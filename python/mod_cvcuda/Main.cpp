/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// NVCV type exports
#include "nvcv/CAPI.hpp"
#include "nvcv/Cache.hpp"
#include "nvcv/ColorSpec.hpp"
#include "nvcv/Container.hpp"
#include "nvcv/DataType.hpp"
#include "nvcv/Definitions.hpp"
#include "nvcv/ExternalBuffer.hpp"
#include "nvcv/Image.hpp"
#include "nvcv/ImageBatch.hpp"
#include "nvcv/ImageFormat.hpp"
#include "nvcv/Rect.hpp"
#include "nvcv/Resource.hpp"
#include "nvcv/Stream.hpp"
#include "nvcv/Tensor.hpp"
#include "nvcv/TensorBatch.hpp"
#include "nvcv/ThreadScope.hpp"

// CV-CUDA Types exports
#include "AdaptiveThresholdType.hpp"
#include "BorderType.hpp"
#include "ChannelManipType.hpp"
#include "ColorConversionCode.hpp"
#include "ConnectivityType.hpp"
#include "FacadeCasterTests.hpp"
#include "HistogramEqMethod.hpp"
#include "InterpolationType.hpp"
#include "LabelType.hpp"
#include "MorphologyType.hpp"
#include "NormType.hpp"
#include "OsdElement.hpp"
#include "PairwiseMatcherType.hpp"
#include "RemapMapValueType.hpp"
#include "RoundMode.hpp"
#include "SIFTFlagType.hpp"
#include "ThresholdType.hpp"

// CV-CUDA Operators exports
#include "operators/Operators.hpp"

#include <common/Binding.hpp>
#include <cvcuda/Version.h>
#include <nvcv/python/ResourceGuard.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

namespace py = nanobind;

namespace {

class SubmissionLeaseProbe final : public cvcudapy::OperatorResource
{
public:
    class Key final : public nvcvpy::IKey
    {
    private:
        size_t doGetHash() const override
        {
            return 0;
        }

        bool doIsCompatible(const nvcvpy::IKey &) const override
        {
            return true;
        }
    };

    void submitSync(nvcvpy::priv::Stream &stream) override
    {
        Resource::submitSync(stream);
        observeSubmitSync();
    }

    void submitSync(nvcvpy::priv::Stream &stream, int deviceId,
                    nvcvpy::priv::StreamOwnerBag *retiredOwners = nullptr) override
    {
        Resource::submitSync(stream, deviceId, retiredOwners);
        observeSubmitSync();
    }

#ifdef NB_FREE_THREADED
    void observeEnqueue()
    {
        observeSubmitSync();
    }
#endif

    void reset()
    {
        m_aEnqueued.store(false, std::memory_order_relaxed);
        m_bReachedSync.store(false, std::memory_order_relaxed);
        m_bSawAEnqueued.store(false, std::memory_order_relaxed);
    }

    inline static thread_local int s_role = 0;

    std::atomic_bool m_aEnqueued{false};
    std::atomic_bool m_bReachedSync{false};
    std::atomic_bool m_bSawAEnqueued{false};

private:
    void observeSubmitSync()
    {
        if (s_role == 2)
        {
            m_bReachedSync.store(true, std::memory_order_release);
            m_bSawAEnqueued.store(m_aEnqueued.load(std::memory_order_acquire), std::memory_order_release);
        }
    }
};

struct ResourceGuardModeProbeCounts
{
    std::atomic_int syncCalls{0};
    std::atomic_int submissionMutexQueries{0};
};

class ResourceGuardModeProbe final : public nvcvpy::priv::Resource
{
public:
    using nvcvpy::priv::Resource::submitSync;

    explicit ResourceGuardModeProbe(ResourceGuardModeProbeCounts &counts)
        : m_counts(counts)
    {
    }

    void submitSync(nvcvpy::priv::Stream &stream, int deviceId,
                    nvcvpy::priv::StreamOwnerBag *retiredOwners = nullptr) override
    {
        m_counts.syncCalls.fetch_add(1, std::memory_order_relaxed);
        Resource::submitSync(stream, deviceId, retiredOwners);
    }

    std::mutex *submissionMutex() const override
    {
        m_counts.submissionMutexQueries.fetch_add(1, std::memory_order_relaxed);
        return &m_submissionMutex;
    }

private:
    ResourceGuardModeProbeCounts &m_counts;
    mutable std::mutex            m_submissionMutex;
};

} // namespace

namespace cvcudapy {

template<>
struct OperatorCacheUsesCompletionLease<::SubmissionLeaseProbe> : std::true_type
{
};

} // namespace cvcudapy

namespace {

// Dedicated failure type for the module self-test probes, so the failure sites
// throw a specific exception rather than the generic std::runtime_error.
class SelfTestError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

// Shared timeout for the probe's cross-thread coordination waits. Kept at
// namespace scope so worker-thread lambdas can use it without a capture.
constexpr std::chrono::seconds kCoordinationTimeout{10};

// The probe's two threads are extracted so each function reads as one thread's
// whole contribution to the interleaving under test.
struct LeaseProbeCoordination
{
    std::mutex              mutex;
    std::condition_variable cv;
    bool                    aReachedCallable = false;
    bool                    bAttemptedRun    = false;
    bool                    allowAEnqueue    = false;
    std::exception_ptr      error;
};

void RunLeaseProbeFirstSubmission(LeaseProbeCoordination &coord, const std::shared_ptr<SubmissionLeaseProbe> &state,
                                  nvcvpy::priv::Stream &stream, std::chrono::seconds timeout)
{
    try
    {
        SubmissionLeaseProbe::s_role = 1;
        cvcudapy::NativeResourceGuard guard(stream);
        guard.add(nvcvpy::LockMode::LOCK_MODE_READWRITE, {*state});
        guard.run(
            [&coord, &state, timeout]
            {
                std::unique_lock lock(coord.mutex);
                coord.aReachedCallable = true;
                coord.cv.notify_all();
                if (!coord.cv.wait_for(lock, timeout, [&coord] { return coord.allowAEnqueue; }))
                {
                    throw SelfTestError("timed out waiting to release the first submission");
                }
                state->m_aEnqueued.store(true, std::memory_order_release);
            });
    }
    catch (...) // NOSONAR: capture all for main-thread rethrow; escaping would call std::terminate.
    {
        std::lock_guard lock(coord.mutex);
        if (!coord.error)
        {
            coord.error = std::current_exception();
        }
        coord.aReachedCallable = true;
        coord.cv.notify_all();
    }
}

void RunLeaseProbeSecondSubmission(LeaseProbeCoordination &coord, const std::shared_ptr<SubmissionLeaseProbe> &reused,
                                   nvcvpy::priv::Stream &stream)
{
    try
    {
        SubmissionLeaseProbe::s_role = 2;
        {
            std::lock_guard lock(coord.mutex);
            coord.bAttemptedRun = true;
        }
        coord.cv.notify_all();

        cvcudapy::NativeResourceGuard guard(stream);
        guard.add(nvcvpy::LockMode::LOCK_MODE_READWRITE, {*reused});
        guard.run(
            [
#ifdef NB_FREE_THREADED
                &reused
#endif
        ]
            {
#ifdef NB_FREE_THREADED
                reused->observeEnqueue();
#endif
            });
    }
    catch (...) // NOSONAR: capture all for main-thread rethrow; escaping would call std::terminate.
    {
        std::lock_guard lock(coord.mutex);
        if (!coord.error)
        {
            coord.error = std::current_exception();
        }
    }
}

void TestTypedOperatorSubmissionLease()
{
    auto state       = cvcudapy::CreateOperatorEx<SubmissionLeaseProbe>();
    auto independent = cvcudapy::CreateOperatorEx<SubmissionLeaseProbe>();
    if (state == independent)
    {
        throw SelfTestError("completion-leased operator cache reused a host-leased state");
    }
    independent.reset();

    // Exercise the submission mutex with two aliases of one explicit lease.
    // Independent cache acquisitions must no longer manufacture this alias.
    auto reused = state;
    state->reset();

    auto streamA = nvcvpy::priv::Stream::Create();
    auto streamB = nvcvpy::priv::Stream::Create();

    LeaseProbeCoordination coord;
    bool                   leaseHeld            = false;
    bool                   coordinationTimedOut = false;

    {
        py::gil_scoped_release release;

        std::thread threadA([&coord, &state, &streamA]
                            { RunLeaseProbeFirstSubmission(coord, state, *streamA, kCoordinationTimeout); });

        {
            std::unique_lock lock(coord.mutex);
            if (!coord.cv.wait_for(lock, kCoordinationTimeout, [&coord] { return coord.aReachedCallable; }))
            {
                coordinationTimedOut = true;
                coord.allowAEnqueue  = true;
            }
        }
        coord.cv.notify_all();

        std::thread threadB;
        if (!coordinationTimedOut)
        {
            threadB
                = std::thread([&coord, &reused, &streamB] { RunLeaseProbeSecondSubmission(coord, reused, *streamB); });

            {
                std::unique_lock lock(coord.mutex);
                if (!coord.cv.wait_for(lock, kCoordinationTimeout, [&coord] { return coord.bAttemptedRun; }))
                {
                    coordinationTimedOut = true;
                }
            }

            std::mutex *submissionMutex = state->submissionMutex();
            leaseHeld                   = submissionMutex != nullptr && !submissionMutex->try_lock();
            if (!leaseHeld && submissionMutex != nullptr)
            {
                submissionMutex->unlock();
            }
        }

        {
            std::lock_guard lock(coord.mutex);
            coord.allowAEnqueue = true;
        }
        coord.cv.notify_all();

        threadA.join();
        if (threadB.joinable())
        {
            threadB.join();
        }
    }
    streamA->sync();
    streamB->sync();

    if (coord.error)
    {
        std::rethrow_exception(coord.error);
    }
    if (coordinationTimedOut)
    {
        throw SelfTestError("operator submission lease probe timed out");
    }
    if (!leaseHeld || !state->m_bReachedSync.load(std::memory_order_acquire)
        || !state->m_bSawAEnqueued.load(std::memory_order_acquire))
    {
        throw SelfTestError("operator submission lease did not cover synchronization through enqueue");
    }

    const SubmissionLeaseProbe *original = state.get();
    state.reset();
    reused.reset();
    auto retired = cvcudapy::CreateOperatorEx<SubmissionLeaseProbe>();
    if (retired.get() != original)
    {
        throw SelfTestError("completion-leased operator state was not reusable after pending release");
    }
}

void TestNativeResourceGuardModes()
{
    auto stream = nvcvpy::priv::Stream::Create();

    auto runCase = [&stream](nvcvpy::LockMode mode, int expectedSyncCalls, int expectedMutexQueries)
    {
        ResourceGuardModeProbeCounts          counts;
        std::weak_ptr<ResourceGuardModeProbe> weakResource;
        {
            auto resource = std::make_shared<ResourceGuardModeProbe>(counts);
            weakResource  = resource;

            cvcudapy::NativeResourceGuard guard(*stream);
            guard.add(mode, {*resource});
            resource.reset();
            guard.run(
                [&weakResource]
                {
                    if (weakResource.expired())
                    {
                        throw SelfTestError("native guard dropped a lifetime owner before submission");
                    }
                });

            if (weakResource.expired())
            {
                throw SelfTestError("native guard dropped a lifetime owner before pending release");
            }
#ifdef NB_FREE_THREADED
            auto resourceState = weakResource.lock()->submissionStates().front()->exportAccess();
            bool prepared      = resourceState.owner == stream->handleOwner();
            if (prepared != (expectedSyncCalls != 0))
                throw SelfTestError("native guard selected the wrong CUDA synchronization set");
            expectedSyncCalls = 0; // Free-threaded guards synchronize allocation state directly.
#endif
        }

        if (counts.syncCalls.load(std::memory_order_relaxed) != expectedSyncCalls
            || counts.submissionMutexQueries.load(std::memory_order_relaxed) != expectedMutexQueries)
        {
            throw SelfTestError("native guard lock mode selected the wrong synchronization set");
        }

        stream->sync();
        nvcvpy::priv::Stream::DrainAllPendingReleases();
        if (!weakResource.expired())
        {
            throw SelfTestError("native guard retained a lifetime owner after GPU retirement");
        }
    };

    runCase(nvcvpy::LockMode::LOCK_MODE_NONE, 0, 0);
    runCase(nvcvpy::LockMode::LOCK_MODE_READWRITE, 1, 1);
}

void RetirePendingReleasesOffGil(const std::shared_ptr<nvcvpy::priv::Stream> &stream)
{
    ResourceGuardModeProbeCounts counts;
    auto                         resource = std::make_shared<ResourceGuardModeProbe>(counts);
    {
        py::gil_scoped_release        release;
        cvcudapy::NativeResourceGuard guard(*stream);
        guard.add(nvcvpy::LockMode::LOCK_MODE_NONE, {*resource});
        guard.run([] { /* no GPU work: this probe exercises retirement only */ });
    }

    stream->sync();
    nvcvpy::priv::Stream::DrainAllPendingReleases();
}

void EnqueueNativeNoop(const std::shared_ptr<nvcvpy::priv::Stream> &stream)
{
    ResourceGuardModeProbeCounts  counts;
    auto                          resource = std::make_shared<ResourceGuardModeProbe>(counts);
    cvcudapy::NativeResourceGuard guard(*stream);
    guard.add(nvcvpy::LockMode::LOCK_MODE_NONE, {*resource});
    guard.run([] { /* no GPU work: this probe leaves one native batch pending */ });
}

} // namespace

NB_MODULE(_cvcuda, m)
{
    // Format and TensorLayout constants, plus Stream.default, are intentionally
    // interned for the process lifetime. Their class/instance reference cycles
    // look like leaks to nanobind's shutdown diagnostic but are required API
    // singletons, so do not emit misleading warnings to application stderr.
    py::set_leak_warnings(false);

    m.attr("__name__")    = "cvcuda";
    m.attr("__version__") = CVCUDA_VERSION_STRING;

    {
        using namespace nvcvpy::priv;

        // Submodule used for additional functionality needed only by tests
        // so that some level of white-box testing is possible.
        //
        // This guarantees a clear separation of public and private APIs.
        // Users are restricted to the public API, allowing us to change the
        // private APIs as needed, without worring in breaking user's code.
        //
        // To retrieve it from inside the Export call, include "Definitions.hpp"
        // and call:
        //    py::module_ internal = m.attr(INTERNAL_SUBMODULE_NAME);
        // Functions and other properties can be then exposed as usual, e.g.
        //    internal.def("foo", &Foo");
        // and accessed in python as you'd expect:
        //    cvcuda.internal.foo()
        m.def_submodule(INTERNAL_SUBMODULE_NAME);

        // Export NVCV core types (Tensor, Image, Format, etc.)
        // Supporting objects
        ExportColorSpec(m);
        ExportImageFormat(m);
        ExportDataType(m);
        ExportRect(m);
        ExportThreadScope(m);
        ExportTensorLayout(m); // Image::cpu/cuda take a TensorLayout — must be registered before Image::Export.

        // Core Entities
        // Registration order matters: (a) base classes must be registered
        // before derived ones (Resource → Container → Tensor/Image/...);
        // (b) when a method signature references another nanobind type, that
        // type must be registered first, or nanobind emits the raw C++
        // typename (e.g. `nvcvpy::priv::Stream`) in stubs and reprs.
        ExportCAPI(m);
        Cache::Export(m);     // exports cache controls; cache metadata stays internal
        Stream::Export(m);    // Resource methods reference Stream
        Resource::Export(m);  // base of Container
        Container::Export(m); // depends on Resource
        ExternalBuffer::Export(m);

        // Objects
        Image::Export(m); // before Tensor (as_tensor takes Image&)
        Tensor::Export(m);
        TensorBatch::Export(m);
        ImageBatchVarShape::Export(m);

        py::module_ test = m.def_submodule("_test");
#ifdef NB_FREE_THREADED
        test.def("native_resourceguard_run",
                 [](Resource &resource, Stream &stream, py::object callable)
                 {
                     cvcudapy::NativeResourceGuard guard(stream);
                     guard.add(nvcvpy::LOCK_MODE_WRITE, {resource});
                     guard.run([&] { callable(); });
                 });
        test.def("try_lock_submission",
                 [](Resource &resource)
                 {
                     SubmissionLocks locks;
                     locks.add(resource.submissionStates(), nvcvpy::LOCK_MODE_READWRITE);
                     return locks.try_lock();
                 });
        test.def("load_external_buffer",
                 [](py::object object)
                 {
                     auto buffer = std::make_shared<ExternalBuffer>();
                     if (!buffer->load(object.ptr()))
                         throw py::type_error("Expected a CUDA buffer");
                     return buffer;
                 });
        test.def("capi_legacy_resource_submit_sync_offset", [] { return offsetof(nvcvpy::CAPI, Resource_SubmitSync); });
#endif
        test.def("resourceguard_destructor_error",
                 []()
                 {
                     nvcvpy::Stream        stream = nvcvpy::Stream::Current();
                     nvcvpy::ResourceGuard guard(stream);

                     PyErr_SetString(PyExc_RuntimeError, "injected ResourceGuard commit failure");
                 });
        test.def("resourceguard_legacy_commit",
                 [](nvcvpy::Stream stream, py::object pyResource)
                 {
                     nvcvpy::Resource      resource(std::move(pyResource));
                     nvcvpy::ResourceGuard guard(stream);
                     guard.add(nvcvpy::LockMode::LOCK_MODE_READ, {resource});
                     guard.commit();
                 });
        test.def("resourceguard_run_adds_first_resource",
                 [](nvcvpy::Stream stream, py::object pyResource)
                 {
                     nvcvpy::Resource      resource(std::move(pyResource));
                     nvcvpy::ResourceGuard guard(stream);
                     guard.run([&guard, &resource] { guard.add(nvcvpy::LockMode::LOCK_MODE_READ, {resource}); });
                 });
        test.def("legacy_resourceguard_noop",
                 [](nvcvpy::Stream stream, py::object pyResource)
                 {
                     // Exercise the public header/C-API path retained for
                     // out-of-tree bindings independently of the native guard.
                     nvcvpy::Resource      resource(std::move(pyResource));
                     nvcvpy::ResourceGuard guard(stream);
                     guard.add(nvcvpy::LockMode::LOCK_MODE_READ, {resource});
                     guard.run([] { /* no GPU work: exercise the legacy commit path */ });
                 });
        test.def("release_dlpack_capsule_off_gil", &ExternalBuffer::ReleaseDLPackCapsuleOffGil);
        ExportFacadeCasterTests(test);
        test.def("native_resourceguard_modes", &TestNativeResourceGuardModes);
        test.def("retire_pending_releases_off_gil", &RetirePendingReleasesOffGil);
        test.def("enqueue_native_noop", &EnqueueNativeNoop);
        test.def("typed_operator_submission_lease", &TestTypedOperatorSubmissionLease);
        test.def("thread_scope_roundtrip", [](ThreadScope scope) { return scope; });
        Cache::ExportTestHooks(test);
        ExportCAPITestHooks(test);
    }

    {
        using namespace cvcudapy;

        // doctag: Non-Operators
        // Operators' auxiliary entities
        ExportAdaptiveThresholdType(m);
        ExportHistogramEqMethod(m);
        ExportBorderType(m);
        ExportBoxBlur(m);
        ExportChannelManipType(m);
        ExportOSD(m);
        ExportColorConversionCode(m);
        ExportConnectivityType(m);
        ExportInterpolationType(m);
        ExportLabelType(m);
        ExportPairwiseMatcherType(m);
        ExportMorphologyType(m);
        ExportNormType(m);
        ExportRemapMapValueType(m);
        ExportRoundMode(m);
        ExportSIFTFlagType(m);
        ExportThresholdType(m);

        // doctag: Operators
        // CV-CUDA Operators
        ExportOpMerge(m);
        ExportOpSplit(m);
        ExportOpTvResize(m);
        ExportOpJpegCompressionDistortion(m);
        ExportOpAdjustHue(m);
        ExportOpAdjustSaturation(m);
        ExportOpAdjustSharpness(m);
        ExportOpAdjustContrast(m);
        ExportOpInvert(m);
        ExportOpSolarize(m);
        ExportOpPosterize(m);
        ExportOpAutoContrast(m);
        ExportOpResizeCropConvertReformat(m);
        ExportOpPairwiseMatcher(m);
        ExportOpLabel(m);
        ExportOpOSD(m);
        ExportOpHistogramEq(m);
        ExportOpAdvCvtColor(m);
        ExportOpSIFT(m);
        ExportOpMinMaxLoc(m);
        ExportOpHistogram(m);
        ExportOpMinAreaRect(m);
        ExportOpBndBox(m);
        ExportOpBoxBlur(m);
        ExportOpBrightnessContrast(m);
        ExportOpColorTwist(m);
        ExportOpHQResize(m);
        ExportOpRemap(m);
        ExportOpCropFlipNormalizeReformat(m);
        ExportOpNonMaximumSuppression(m);
        ExportOpReformat(m);
        ExportOpResize(m);
        ExportOpCustomCrop(m);
        ExportOpNormalize(m);
        ExportOpConvertTo(m);
        ExportOpPadAndStack(m);
        ExportOpCopyMakeBorder(m);
        ExportOpRotate(m);
        ExportOpErase(m);
        ExportOpGaussian(m);
        ExportOpMedianBlur(m);
        ExportOpLaplacian(m);
        ExportOpAverageBlur(m);
        ExportOpConv2D(m);
        ExportOpBilateralFilter(m);
        ExportOpJointBilateralFilter(m);
        ExportOpCenterCrop(m);
        ExportOpWarpAffine(m);
        ExportOpWarpPerspective(m);
        ExportOpChannelReorder(m);
        ExportOpMorphology(m);
        ExportOpFlip(m);
        ExportOpCvtColor(m);
        ExportOpComposite(m);
        ExportOpGammaContrast(m);
        ExportOpPillowResize(m);
        ExportOpThreshold(m);
        ExportOpAdaptiveThreshold(m);
        ExportOpRandomResizedCrop(m);
        ExportOpGaussianNoise(m);
        ExportOpInpaint(m);
        ExportOpStack(m);
        ExportOpFindHomography(m);
        ExportOpCLAHE(m);
    }
}
