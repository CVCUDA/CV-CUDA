/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda_runtime_api.h>
// cupti.h already pulls in cupti_runtime_cbid.h for the CUPTI_RUNTIME_TRACE_CBID_*
// values used below. Including that header again directly breaks toolkits that ship
// an unguarded copy of it, such as JetPack's, where the second inclusion redefines
// CUpti_runtime_api_trace_cbid_enum.
#include <cupti.h>
#include <dlfcn.h>
#include <dlpack/dlpack.h>
#include <nvcv/Image.h>
#include <nvcv/Tensor.h>
#include <pthread.h>

#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <unordered_set>

extern "C" int32_t cvcudaFlipSubmit(void *handle, void *stream, void *input, void *output, int32_t flipCode);
extern "C" int32_t cvcudaFlipVarShapeSubmit(void *handle, void *stream, void *input, void *output, void *flipCode);
extern "C" int32_t cvcudaFindHomographySubmit(void *handle, void *stream, void *srcPts, void *dstPts, void *models);
extern "C" int32_t cvcudaMorphologySubmit(void *handle, void *stream, void *input, void *output, void *workspace,
                                          int32_t morphType, int32_t maskWidth, int32_t maskHeight, int32_t anchorX,
                                          int32_t anchorY, int32_t iteration, int32_t borderMode);
extern "C" int32_t cvcudaMorphologyVarShapeSubmit(void *handle, void *stream, void *input, void *output,
                                                  void *workspace, int32_t morphType, void *masks, void *anchors,
                                                  int32_t iteration, int32_t borderMode);
extern "C" int32_t cvcudaStackSubmit(void *handle, void *stream, void *input, void *output);
extern "C" int32_t cvcudaStackVarShapeSubmit(void *handle, void *stream, void *input, void *output);
extern "C" void    cvcudaTestArmFlipGate();
extern "C" int     cvcudaTestWaitForFirstFlip(int timeoutMilliseconds);
extern "C" int     cvcudaTestWaitForSecondFlip(int timeoutMilliseconds);
extern "C" int     cvcudaTestFlipOverlapObserved();
extern "C" void    cvcudaTestReleaseFlip();
extern "C" void    cvcudaTestReleaseFlipAndObserveSecond();
extern "C" void    cvcudaTestArmDetachedFlipGate();
extern "C" void    cvcudaTestArmFlipReturnProbe();
extern "C" int     cvcudaTestWaitForFlipReturn(int timeoutMilliseconds);
extern "C" void    cvcudaTestArmLifetimeGate();
extern "C" int     cvcudaTestWaitForLifetimeGate(int timeoutMilliseconds);
extern "C" int     cvcudaTestWaitForLifetimeSubmit(int timeoutMilliseconds);
extern "C" int     cvcudaTestLifetimeWorkPending();
extern "C" void    cvcudaTestReleaseLifetimeGate();
extern "C" int     cvcudaTestTensorDestroyedBeforeLifetimeRelease();
extern "C" void    cvcudaTestStopResourceDestructionTracking();
extern "C" int     cvcudaTestTrackedTensorCount();
extern "C" int     cvcudaTestDestroyedTensorCount();
extern "C" void    cvcudaTestArmReaderEntryGate(int targetCalls);
extern "C" void    cvcudaTestReachReaderEntry();
extern "C" int     cvcudaTestWaitForReaderEntry(int timeoutMilliseconds);
extern "C" void    cvcudaTestReleaseReaderEntryGate();
extern "C" void    cvcudaTestArmNextBusyMutexAttempt();
extern "C" int     cvcudaTestWaitForBusyMutexAttempt(int timeoutMilliseconds);
extern "C" void    cvcudaTestReleaseBusyMutexAttempt();
extern "C" void    cvcudaTestArmBatchMutation(int action);
extern "C" int     cvcudaTestWaitForBatchPush(int timeoutMilliseconds);
extern "C" int     cvcudaTestWaitForStackSubmit(int timeoutMilliseconds);
extern "C" void    cvcudaTestReleaseBatchPush();
extern "C" void    cvcudaTestArmStackGate();
extern "C" void    cvcudaTestReleaseStackGate();
extern "C" void    cvcudaTestArmConcurrentStackGate();
extern "C" int     cvcudaTestWaitForFirstConcurrentStack(int timeoutMilliseconds);
extern "C" int     cvcudaTestWaitForSecondConcurrentStack(int timeoutMilliseconds);
extern "C" void    cvcudaTestReleaseConcurrentStack();
extern "C" void    cvcudaTestReleaseConcurrentStackAndObserveSecond();
extern "C" void    cvcudaTestArmImageBatchMetadataGate(int detachFirstWaiter);
extern "C" int     cvcudaTestWaitForFirstImageBatchMetadata(int timeoutMilliseconds);
extern "C" int     cvcudaTestWaitForSecondImageBatchMetadata(int timeoutMilliseconds);
extern "C" void    cvcudaTestReleaseImageBatchMetadata();
extern "C" void    cvcudaTestReleaseImageBatchMetadataAndObserveSecond();
extern "C" void    cvcudaTestArmGpuGate();
extern "C" int     cvcudaTestQueueGpuGate(void *stream);
extern "C" int     cvcudaTestWaitForGpuGate(int timeoutMilliseconds);
extern "C" void    cvcudaTestReleaseGpuGate();
extern "C" void    cvcudaTestArmEventRecordGate();
extern "C" void    cvcudaTestArmEventRecordEntryGate();
extern "C" int     cvcudaTestWaitForEventRecordCount(int targetCalls, int timeoutMilliseconds);
extern "C" void    cvcudaTestReleaseEventRecordGate();
extern "C" void    cvcudaTestArmDeviceSynchronizeGate();
extern "C" int     cvcudaTestWaitForDeviceSynchronize(int timeoutMilliseconds);
extern "C" int     cvcudaTestWaitForDeviceSynchronizeCalls(int calls, int timeoutMilliseconds);
extern "C" void    cvcudaTestReleaseDeviceSynchronizeGate();
extern "C" void    cvcudaTestArmGpuSignalGate();
extern "C" int     cvcudaTestQueueGpuSignal(void *stream);
extern "C" int     cvcudaTestWaitForGpuSignal(int timeoutMilliseconds);
extern "C" void    cvcudaTestReleaseGpuSignalGate();
extern "C" void    cvcudaTestArmImageCpuCopyFailure();
extern "C" int     cvcudaTestWaitForImageCpuFirstCopy(int timeoutMilliseconds);
extern "C" int     cvcudaTestWaitForImageCpuCopyFailure(int timeoutMilliseconds);
extern "C" int     cvcudaTestWaitForImageCpuCleanupSync(int timeoutMilliseconds);
extern "C" int     cvcudaTestImageCpuCopyPending();
extern "C" int     cvcudaTestImageCpuCopyApi();
extern "C" void    cvcudaTestReleaseImageCpuFirstCopy();
extern "C" void    cvcudaTestArmCriticalSectionHold();
extern "C" int     cvcudaTestWaitForCriticalSectionHold(int timeoutMilliseconds);
extern "C" void    cvcudaTestHoldCriticalSection(void *targetObject);
extern "C" void    cvcudaTestReleaseCriticalSectionHold();
extern "C" void    cvcudaTestArmDelayedFlipRelease();
extern "C" int     cvcudaTestWaitForDelayedFlipRelease(int timeoutMilliseconds);
extern "C" int     cvcudaTestDelayedFlipReleaseReached();
extern "C" void    cvcudaTestReleaseFlipAfterDelay(int delayMilliseconds);
extern "C" void    cvcudaTestArmFindHomographyGate();
extern "C" int     cvcudaTestWaitForFirstFindHomography(int timeoutMilliseconds);
extern "C" int     cvcudaTestWaitForSecondFindHomography(int timeoutMilliseconds);
extern "C" void    cvcudaTestReleaseFindHomographyAndObserveSecond();
extern "C" void    cvcudaTestArmMorphologyGate();
extern "C" int     cvcudaTestWaitForFirstMorphology(int timeoutMilliseconds);
extern "C" int     cvcudaTestWaitForSecondMorphology(int timeoutMilliseconds);
extern "C" void    cvcudaTestReleaseMorphologyAndObserveSecond();
extern "C" int32_t nvcvTensorBatchPushTensors(void *handle, void *const *tensors, int32_t numTensors);
extern "C" int32_t nvcvImageBatchVarShapePushImages(void *handle, void *const *images, int32_t numImages);
extern "C" int32_t nvcvTensorBatchPopTensors(void *handle, int32_t numTensors);
extern "C" int32_t nvcvTensorBatchClear(void *handle);
extern "C" int32_t nvcvTensorBatchSetTensors(void *handle, int32_t index, void *const *tensors, int32_t numTensors);
extern "C" int32_t nvcvImageBatchVarShapePopImages(void *handle, int32_t numImages);
extern "C" int32_t nvcvImageBatchVarShapeClear(void *handle);
extern "C" int32_t nvcvImageBatchVarShapeGetMaxSize(void *handle, int32_t *maxWidth, int32_t *maxHeight);
extern "C" int32_t nvcvImageBatchVarShapeGetUniqueFormat(void *handle, void *format);
extern "C" void   *cvcudaTestTakeDLPackManager(void *capsule, int versioned);
extern "C" int     cvcudaTestDeleteDLPackManagerOnNativeThread(void *managedTensor, int versioned);
extern "C" int     cvcudaTestNativeDLPackDeleteThreadWasAttached();
extern "C" void   *PyCapsule_GetPointer(void *capsule, const char *name);
extern "C" int     PyCapsule_SetName(void *capsule, const char *name);
extern "C" int     PyGILState_Check();
extern "C" void   *PyEval_SaveThread();
extern "C" void    PyEval_RestoreThread(void *threadState);

namespace {

// Opaque stand-ins for pointers that arrive through the probe's C entry points.
// Distinct incomplete types keep the handle kinds apart in the C++ helpers below
// while staying ABI-identical to the "void *" those entry points receive; the
// concrete type is recovered by a cast only where one is actually needed.
struct CvcudaOperator;

// Either a DLManagedTensor or a DLManagedTensorVersioned, selected by a
// companion "versioned" flag that travels with every such pointer.
struct DLPackManager;

using FlipSubmit           = int32_t (*)(void *, void *, void *, void *, int32_t);
using FlipVarShapeSubmit   = int32_t (*)(void *, void *, void *, void *, void *);
using FindHomographySubmit = int32_t (*)(void *, void *, void *, void *, void *);
using MorphologySubmit     = int32_t (*)(void *, void *, void *, void *, void *, int32_t, int32_t, int32_t, int32_t,
                                     int32_t, int32_t, int32_t);
using MorphologyVarShapeSubmit
    = int32_t (*)(void *, void *, void *, void *, void *, int32_t, void *, void *, int32_t, int32_t);
using StackSubmit               = int32_t (*)(void *, void *, void *, void *);
using BatchPush                 = int32_t (*)(void *, void *const *, int32_t);
using BatchPop                  = int32_t (*)(void *, int32_t);
using BatchClear                = int32_t (*)(void *);
using BatchSet                  = int32_t (*)(void *, int32_t, void *const *, int32_t);
using ImageBatchGetMaxSize      = int32_t (*)(void *, int32_t *, int32_t *);
using ImageBatchGetUniqueFormat = int32_t (*)(void *, void *);
using TensorDecRef              = NVCVStatus (*)(NVCVTensorHandle, int *);
using TensorRefCount            = NVCVStatus (*)(NVCVTensorHandle, int *);
using TensorExportData          = NVCVStatus (*)(NVCVTensorHandle, NVCVTensorData *);
using ImageExportData           = NVCVStatus (*)(NVCVImageHandle, NVCVImageData *);
using CriticalSectionBegin      = void (*)(void *, void *);
using CriticalSectionEnd        = void (*)(void *);
using PthreadMutexTryLock       = int (*)(pthread_mutex_t *);
using ThreadStateGet            = void *(*)();

ThreadStateGet RealThreadStateGet()
{
    static ThreadStateGet get = []
    {
        auto function = reinterpret_cast<ThreadStateGet>(dlsym(RTLD_DEFAULT, "PyThreadState_GetUnchecked"));
        if (function == nullptr)
        {
            function = reinterpret_cast<ThreadStateGet>(dlsym(RTLD_DEFAULT, "_PyThreadState_UncheckedGet"));
        }
        if (function == nullptr)
        {
            std::abort();
        }
        return function;
    }();
    return get;
}

struct EntryGate
{
    std::mutex              mutex;
    std::condition_variable condition;
    bool                    armed       = false;
    bool                    released    = false;
    int                     targetCalls = 0;
    int                     calls       = 0;
};

struct Gate
{
    std::mutex              mutex;
    std::condition_variable condition;
    bool                    armed                      = false;
    bool                    firstReached               = false;
    bool                    secondReached              = false;
    bool                    secondReachedBeforeRelease = false;
    bool                    secondFinished             = false;
    bool                    released                   = false;
    int                     calls                      = 0;
};

struct ImageCpuCopyGate
{
    std::mutex              mutex;
    std::condition_variable condition;
    bool                    armed              = false;
    bool                    firstReached       = false;
    bool                    secondReached      = false;
    bool                    cleanupSyncReached = false;
    bool                    released           = false;
    bool                    pending            = false;
    int                     calls              = 0;
    int                     api                = 0;
    std::thread::id         copyFailureThread;
};

Gate             g_flipGate;
Gate             g_findHomographyGate;
Gate             g_morphologyGate;
Gate             g_concurrentStackGate;
Gate             g_imageBatchMetadataGate;
EntryGate        g_readerEntryGate;
EntryGate        g_stackGate;
EntryGate        g_gpuGate;
EntryGate        g_eventRecordGate;
std::atomic_bool g_eventRecordAtEntry{false};
EntryGate        g_deviceSynchronizeGate;
EntryGate        g_flipReturnGate;
EntryGate        g_criticalSectionHoldGate;
EntryGate        g_busyMutexAttemptGate;
ImageCpuCopyGate g_imageCpuCopyGate;
bool             g_detachFirstFlipWaiter               = false;
bool             g_detachFirstImageBatchMetadataWaiter = false;

std::atomic_int g_nativeDLPackThreadAttached = -1;

std::atomic_bool      g_probeNextBusyMutexAttempt = false;
std::atomic_uintptr_t g_busyMutexThread           = 0;

std::mutex  g_gpuSignalMutex;
cudaEvent_t g_gpuSignalEvent = nullptr;

std::mutex              g_delayedFlipReleaseMutex;
std::condition_variable g_delayedFlipReleaseCondition;
bool                    g_delayedFlipReleaseStarted = false;
bool                    g_delayedFlipReleaseReached = false;

std::mutex                 g_lifetimeMutex;
std::condition_variable    g_lifetimeCondition;
bool                       g_lifetimeArmed                  = false;
bool                       g_lifetimeReached                = false;
bool                       g_lifetimeRelease                = false;
bool                       g_lifetimeSubmitReturned         = false;
cudaEvent_t                g_lifetimeGateEvent              = nullptr;
bool                       g_lifetimeActive                 = false;
bool                       g_destroyedBeforeLifetimeRelease = false;
bool                       g_resourceTracking               = false;
bool                       g_trackOutputResources           = false;
std::unordered_set<void *> g_trackedTensors;
std::unordered_set<void *> g_destroyedTensors;

std::mutex              g_batchMutex;
std::condition_variable g_batchCondition;
int                     g_armedBatchAction  = -1;
bool                    g_batchMutationSeen = false;
bool                    g_releaseBatch      = false;

std::mutex              g_stackMutex;
std::condition_variable g_stackCondition;
bool                    g_stackSubmitReached = false;

template<class F>
F ResolveLibrary(const char *library, const char *name)
{
    void *handle = dlopen(library, RTLD_LAZY | RTLD_NOLOAD);
    if (handle == nullptr)
    {
        std::abort();
    }
    auto function = reinterpret_cast<F>(dlsym(handle, name));
    if (function == nullptr)
    {
        std::abort();
    }
    return function;
}

CriticalSectionBegin RealCriticalSectionBegin()
{
    // CPython may export this from the executable rather than libpython, which
    // RTLD_NEXT skips when this probe is preloaded after the main program.
    static auto begin = reinterpret_cast<CriticalSectionBegin>(dlsym(RTLD_DEFAULT, "PyCriticalSection_Begin"));
    if (begin == nullptr)
    {
        std::abort();
    }
    return begin;
}

CriticalSectionEnd RealCriticalSectionEnd()
{
    static auto end = reinterpret_cast<CriticalSectionEnd>(dlsym(RTLD_DEFAULT, "PyCriticalSection_End"));
    if (end == nullptr)
    {
        std::abort();
    }
    return end;
}

PthreadMutexTryLock RealPthreadMutexTryLock()
{
    static auto lock = reinterpret_cast<PthreadMutexTryLock>(dlsym(RTLD_NEXT, "pthread_mutex_trylock"));
    if (lock == nullptr)
    {
        std::abort();
    }
    return lock;
}

std::chrono::duration<double> ProbeGateTimeout()
{
    static const double scale = []
    {
        const char *configured = std::getenv("CVCUDA_TEST_TIMEOUT_SCALE");
        if (configured == nullptr)
        {
            return 1.0;
        }

        errno        = 0;
        char  *end   = nullptr;
        double value = std::strtod(configured, &end);
        if (errno != 0 || end == configured || *end != '\0' || !std::isfinite(value) || value <= 0)
        {
            std::fprintf(stderr, "FreeThreadingProbe: invalid CVCUDA_TEST_TIMEOUT_SCALE=%s\n", configured);
            std::fflush(stderr);
            std::abort();
        }
        return value;
    }();
    return std::chrono::seconds(20) * scale;
}

void ReportProbeTimeout(const char *gateName)
{
    auto timeout = std::chrono::duration_cast<std::chrono::milliseconds>(ProbeGateTimeout());
    std::fprintf(stderr, "FreeThreadingProbe: %s timed out after %lld ms\n", gateName,
                 static_cast<long long>(timeout.count()));
    std::fflush(stderr);
}

[[noreturn]] void AbortProbeTimeout(const char *gateName)
{
    ReportProbeTimeout(gateName);
    std::abort();
}

bool WaitForGateState(Gate &gate, int timeoutMilliseconds, bool Gate::*state)
{
    std::unique_lock lock(gate.mutex);
    return gate.condition.wait_for(lock, std::chrono::milliseconds(timeoutMilliseconds),
                                   [&gate, state] { return gate.*state; });
}

void ArmGate(Gate &gate)
{
    std::lock_guard lock(gate.mutex);
    gate.armed                      = true;
    gate.firstReached               = false;
    gate.secondReached              = false;
    gate.secondReachedBeforeRelease = false;
    gate.secondFinished             = false;
    gate.released                   = false;
    gate.calls                      = 0;
}

void ReleaseGate(Gate &gate, bool observeSecond = false)
{
    {
        std::lock_guard lock(gate.mutex);
        gate.released = true;
        // Two-sided exclusion tests keep the gate observable until the delayed
        // call proves that it can reach the intercepted submission path.
        gate.armed = observeSecond && gate.calls == 1;
    }
    gate.condition.notify_all();
}

template<class Submit>
int32_t SubmitThroughGate(Gate &gate, const char *gateName, cudaStream_t stream, Submit &&submit)
{
    std::unique_lock lock(gate.mutex);
    if (!gate.armed)
    {
        lock.unlock();
        return submit();
    }

    ++gate.calls;
    int call = gate.calls;
    if (call == 1)
    {
        gate.firstReached = true;
        gate.condition.notify_all();
        if (!gate.condition.wait_for(lock, ProbeGateTimeout(),
                                     [&gate] { return gate.released && (!gate.secondReached || gate.secondFinished); }))
        {
            AbortProbeTimeout(gateName);
        }
        lock.unlock();
        return submit();
    }

    if (call == 2)
    {
        gate.secondReachedBeforeRelease |= !gate.released;
        gate.secondReached = true;
        gate.condition.notify_all();
        lock.unlock();
        int32_t status = submit();
        if (status == 0 && cudaStreamSynchronize(stream) != cudaSuccess)
        {
            std::abort();
        }
        lock.lock();
        gate.secondFinished = true;
        gate.armed          = false;
        lock.unlock();
        gate.condition.notify_all();
        return status;
    }

    lock.unlock();
    return submit();
}

template<class Call>
int32_t CallThroughGate(Gate &gate, const char *gateName, bool detachFirstWaiter, Call &&call)
{
    std::unique_lock lock(gate.mutex);
    if (!gate.armed)
    {
        lock.unlock();
        return call();
    }

    ++gate.calls;
    int callNumber = gate.calls;
    if (callNumber == 1)
    {
        gate.firstReached = true;
        gate.condition.notify_all();
        void *threadState = detachFirstWaiter ? PyEval_SaveThread() : nullptr;
        if (!gate.condition.wait_for(lock, ProbeGateTimeout(),
                                     [&gate] { return gate.released && (!gate.secondReached || gate.secondFinished); }))
        {
            AbortProbeTimeout(gateName);
        }
        lock.unlock();
        int32_t status = call();
        if (threadState != nullptr)
        {
            PyEval_RestoreThread(threadState);
        }
        return status;
    }

    if (callNumber == 2)
    {
        gate.secondReachedBeforeRelease |= !gate.released;
        gate.secondReached = true;
        gate.condition.notify_all();
        lock.unlock();
        int32_t status = call();
        lock.lock();
        gate.secondFinished = true;
        gate.armed          = false;
        lock.unlock();
        gate.condition.notify_all();
        return status;
    }
    lock.unlock();
    return call();
}

void ArmEntryGate(EntryGate &gate, int targetCalls)
{
    std::lock_guard lock(gate.mutex);
    gate.armed       = true;
    gate.released    = false;
    gate.targetCalls = targetCalls;
    gate.calls       = 0;
}

void ReachEntryGate(EntryGate &gate, const char *gateName, bool detachWaiter = false)
{
    std::unique_lock lock(gate.mutex);
    if (!gate.armed)
    {
        return;
    }
    ++gate.calls;
    gate.condition.notify_all();
    void *threadState
        = detachWaiter && PyGILState_Check() && RealThreadStateGet()() != nullptr ? PyEval_SaveThread() : nullptr;
    bool released = gate.condition.wait_for(lock, ProbeGateTimeout(), [&gate] { return gate.released; });
    lock.unlock();
    if (threadState != nullptr)
    {
        PyEval_RestoreThread(threadState);
    }
    if (!released)
    {
        AbortProbeTimeout(gateName);
    }
}

void ObserveEntryGate(EntryGate &gate)
{
    std::lock_guard lock(gate.mutex);
    if (gate.armed)
    {
        ++gate.calls;
        gate.condition.notify_all();
    }
}

int WaitForEntryGate(EntryGate &gate, int timeoutMilliseconds)
{
    std::unique_lock lock(gate.mutex);
    bool             reached = gate.condition.wait_for(
                    lock, std::chrono::milliseconds(timeoutMilliseconds),
                    [&gate] { return gate.armed && gate.targetCalls > 0 && gate.calls >= gate.targetCalls; });
    return reached ? 1 : 0;
}

void ReleaseEntryGate(EntryGate &gate)
{
    {
        std::lock_guard lock(gate.mutex);
        gate.released    = true;
        gate.armed       = false;
        gate.calls       = 0;
        gate.targetCalls = 0;
    }
    gate.condition.notify_all();
}

int32_t PauseBatchMutation(int action, int32_t status)
{
    if (status != 0)
    {
        return status;
    }

    std::unique_lock lock(g_batchMutex);
    if (g_armedBatchAction != action)
    {
        return status;
    }
    g_armedBatchAction  = -1;
    g_batchMutationSeen = true;
    g_batchCondition.notify_all();
    if (!g_batchCondition.wait_for(lock, ProbeGateTimeout(), [] { return g_releaseBatch; }))
    {
        AbortProbeTimeout("batch-mutation gate");
    }
    return status;
}

bool PrepareFlipSubmission(NVCVTensorHandle input, NVCVTensorHandle output)
{
    bool armed;
    {
        std::lock_guard lock(g_lifetimeMutex);
        if (g_resourceTracking)
        {
            g_trackedTensors.insert(input);
            if (g_trackOutputResources)
            {
                g_trackedTensors.insert(output);
            }
        }
        armed           = g_lifetimeArmed;
        g_lifetimeArmed = false;
    }
    return armed;
}

void QueueLifetimeWork(cudaStream_t stream, NVCVTensorHandle output)
{
    static TensorExportData exportData = ResolveLibrary<TensorExportData>("libnvcv_types.so.0", "nvcvTensorExportData");
    // The matching test allocates a contiguous 4096x4096x4 U8 output. Repeated
    // writes keep that exact allocation in active GPU use long enough for the
    // submitting Python thread to exit and the lifetime counters to be sampled.
    constexpr size_t        kBytes       = 64ULL * 1024 * 1024;
    constexpr int           kRepetitions = 8192;

    NVCVTensorData data{};
    if (exportData(output, &data) != NVCV_SUCCESS || data.bufferType != NVCV_TENSOR_BUFFER_STRIDED_CUDA)
    {
        std::abort();
    }
    for (int i = 0; i < kRepetitions; ++i)
    {
        if (cudaMemsetAsync(data.buffer.strided.basePtr, i, kBytes, stream) != cudaSuccess)
        {
            std::abort();
        }
    }
    if (cudaEventRecord(g_lifetimeGateEvent, stream) != cudaSuccess)
    {
        std::abort();
    }
    {
        std::lock_guard lock(g_lifetimeMutex);
        g_lifetimeReached        = true;
        g_lifetimeSubmitReturned = true;
    }
    g_lifetimeCondition.notify_all();
}

int32_t SubmitRealFlip(CvcudaOperator *handle, cudaStream_t stream, NVCVTensorHandle input, NVCVTensorHandle output,
                       int32_t flipCode)
{
    static FlipSubmit submit             = ResolveLibrary<FlipSubmit>("libcvcuda.so.0", "cvcudaFlipSubmit");
    bool              lifetimeSubmission = PrepareFlipSubmission(input, output);
    int32_t           status             = submit(handle, stream, input, output, flipCode);
    if (lifetimeSubmission && status == 0)
    {
        QueueLifetimeWork(stream, output);
    }
    ObserveEntryGate(g_flipReturnGate);
    return status;
}

bool IsImageCpuSyncCopy(CUpti_CallbackId cbid)
{
    return cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2D_v3020
        || cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2D_ptds_v7000;
}

bool IsImageCpuAsyncCopy(CUpti_CallbackId cbid)
{
    return cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DAsync_v3020
        || cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DAsync_ptsz_v7000;
}

void ObserveImageCpuCopy(CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const CUpti_CallbackData *data)
{
    bool isSyncCopy   = IsImageCpuSyncCopy(cbid);
    bool isAsyncCopy  = IsImageCpuAsyncCopy(cbid);
    bool isStreamSync = cbid == CUPTI_RUNTIME_TRACE_CBID_cudaStreamSynchronize_v3020
                     || cbid == CUPTI_RUNTIME_TRACE_CBID_cudaStreamSynchronize_ptsz_v7000;
    if (domain != CUPTI_CB_DOMAIN_RUNTIME_API || (!isSyncCopy && !isAsyncCopy && !isStreamSync))
    {
        return;
    }

    std::unique_lock lock(g_imageCpuCopyGate.mutex);
    if (isStreamSync)
    {
        if (data->callbackSite == CUPTI_API_ENTER && g_imageCpuCopyGate.armed && g_imageCpuCopyGate.api == 2
            && g_imageCpuCopyGate.secondReached && g_imageCpuCopyGate.pending
            && g_imageCpuCopyGate.copyFailureThread == std::this_thread::get_id())
        {
            g_imageCpuCopyGate.cleanupSyncReached = true;
            g_imageCpuCopyGate.condition.notify_all();
        }
        return;
    }
    if (data->callbackSite == CUPTI_API_EXIT && isSyncCopy && g_imageCpuCopyGate.api == 1
        && g_imageCpuCopyGate.calls == 1)
    {
        g_imageCpuCopyGate.pending = false;
        g_imageCpuCopyGate.condition.notify_all();
        return;
    }
    if (!g_imageCpuCopyGate.armed)
    {
        return;
    }

    if (data->callbackSite == CUPTI_API_ENTER)
    {
        ++g_imageCpuCopyGate.calls;
        int call = g_imageCpuCopyGate.calls;
        if (call == 1)
        {
            g_imageCpuCopyGate.api          = isSyncCopy ? 1 : 2;
            g_imageCpuCopyGate.pending      = true;
            g_imageCpuCopyGate.firstReached = true;
            g_imageCpuCopyGate.condition.notify_all();
            if (isSyncCopy
                && !g_imageCpuCopyGate.condition.wait_for(lock, ProbeGateTimeout(),
                                                          [] { return g_imageCpuCopyGate.released; }))
            {
                AbortProbeTimeout("image CPU-copy gate");
            }
        }
        else if (call == 2)
        {
            g_imageCpuCopyGate.secondReached     = true;
            g_imageCpuCopyGate.copyFailureThread = std::this_thread::get_id();
            g_imageCpuCopyGate.condition.notify_all();
        }
    }
}

void CUPTIAPI ObserveCudaRuntime(void *, CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const void *callbackData);

[[noreturn]] void AbortCuptiFailure(const char *operation, CUptiResult status)
{
    const char *description = nullptr;
    if (cuptiGetResultString(status, &description) != CUPTI_SUCCESS || description == nullptr)
    {
        description = "unknown CUPTI error";
    }
    std::fprintf(stderr, "FreeThreadingProbe: %s failed: %s (%d)\n", operation, description, static_cast<int>(status));
    std::fflush(stderr);
    std::abort();
}

CUpti_SubscriberHandle CuptiSubscriber()
{
    static std::once_flag         enabled;
    static CUpti_SubscriberHandle subscriber{};
    std::call_once(enabled,
                   []
                   {
                       CUptiResult status = cuptiSubscribe(&subscriber, ObserveCudaRuntime, nullptr);
                       if (status != CUPTI_SUCCESS)
                       {
                           AbortCuptiFailure("cuptiSubscribe", status);
                       }
                   });
    return subscriber;
}

void EnableCuptiCallback(CUpti_CallbackId callback)
{
    CUptiResult status = cuptiEnableCallback(1, CuptiSubscriber(), CUPTI_CB_DOMAIN_RUNTIME_API, callback);
    if (status != CUPTI_SUCCESS)
    {
        AbortCuptiFailure("cuptiEnableCallback", status);
    }
}

void EnableImageCpuCopyProbe()
{
    static std::once_flag enabled;
    std::call_once(enabled,
                   []
                   {
                       constexpr std::array<CUpti_CallbackId, 6> callbacks{
                           CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2D_v3020,
                           CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2D_ptds_v7000,
                           CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DAsync_v3020,
                           CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy2DAsync_ptsz_v7000,
                           CUPTI_RUNTIME_TRACE_CBID_cudaStreamSynchronize_v3020,
                           CUPTI_RUNTIME_TRACE_CBID_cudaStreamSynchronize_ptsz_v7000,
                       };
                       for (CUpti_CallbackId callback : callbacks)
                       {
                           EnableCuptiCallback(callback);
                       }
                   });
}

void ObserveEventRecord(CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const CUpti_CallbackData *data)
{
    if (domain != CUPTI_CB_DOMAIN_RUNTIME_API
        || (cbid != CUPTI_RUNTIME_TRACE_CBID_cudaEventRecord_v3020
            && cbid != CUPTI_RUNTIME_TRACE_CBID_cudaEventRecord_ptsz_v7000))
    {
        return;
    }

    if (data->callbackSite == (g_eventRecordAtEntry.load() ? CUPTI_API_ENTER : CUPTI_API_EXIT))
    {
        ReachEntryGate(g_eventRecordGate, "event-record gate", true);
    }
}

void EnableEventRecordProbe()
{
    static std::once_flag enabled;
    std::call_once(enabled,
                   []
                   {
                       constexpr std::array<CUpti_CallbackId, 2> callbacks{
                           CUPTI_RUNTIME_TRACE_CBID_cudaEventRecord_v3020,
                           CUPTI_RUNTIME_TRACE_CBID_cudaEventRecord_ptsz_v7000,
                       };
                       for (CUpti_CallbackId callback : callbacks)
                       {
                           EnableCuptiCallback(callback);
                       }
                   });
}

void ObserveDeviceSynchronize(CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const CUpti_CallbackData *data)
{
    if (domain != CUPTI_CB_DOMAIN_RUNTIME_API || cbid != CUPTI_RUNTIME_TRACE_CBID_cudaDeviceSynchronize_v3020)
    {
        return;
    }

    if (data->callbackSite == CUPTI_API_ENTER)
    {
        ReachEntryGate(g_deviceSynchronizeGate, "device-synchronize gate", true);
    }
}

void EnableDeviceSynchronizeProbe()
{
    static std::once_flag enabled;
    std::call_once(enabled, [] { EnableCuptiCallback(CUPTI_RUNTIME_TRACE_CBID_cudaDeviceSynchronize_v3020); });
}

void CUPTIAPI ObserveCudaRuntime(
    void *, // NOSONAR: CUPTI's callback ABI requires an opaque user-data pointer.
    CUpti_CallbackDomain domain, CUpti_CallbackId cbid,
    const void *callbackData) // NOSONAR: CUPTI's callback ABI requires an opaque callback-data pointer.
{
    const auto *data = static_cast<const CUpti_CallbackData *>(callbackData);
    ObserveImageCpuCopy(domain, cbid, data);
    ObserveEventRecord(domain, cbid, data);
    ObserveDeviceSynchronize(domain, cbid, data);
}

void CUDART_CB ReachGpuGate(void *)
{
    ReachEntryGate(g_gpuGate, "GPU callback gate");
}

} // namespace

extern "C" int pthread_mutex_trylock(pthread_mutex_t *mutex)
{
    int result = RealPthreadMutexTryLock()(mutex);
    // This thread-local window observes the contender after it has completed
    // the state snapshot that precedes UniqueLockDetached's first try_lock.
    if (result == EBUSY && g_probeNextBusyMutexAttempt.load(std::memory_order_acquire)
        && g_busyMutexThread.load(std::memory_order_relaxed) == pthread_self()
        && g_probeNextBusyMutexAttempt.exchange(false, std::memory_order_acq_rel))
    {
        ObserveEntryGate(g_busyMutexAttemptGate);
    }
    return result;
}

extern "C" int32_t cvcudaFlipSubmit(void *handle, void *stream, void *input, void *output, int32_t flipCode)
{
    auto *op                = static_cast<CvcudaOperator *>(handle);
    auto  cudaStream        = static_cast<cudaStream_t>(stream);
    auto *inTensor          = static_cast<NVCVTensorHandle>(input);
    auto *outTensor         = static_cast<NVCVTensorHandle>(output);
    bool  detachFirstWaiter = g_detachFirstFlipWaiter && PyGILState_Check() && RealThreadStateGet()() != nullptr;

    return CallThroughGate(g_flipGate, "tensor flip gate", detachFirstWaiter,
                           [op, cudaStream, inTensor, outTensor, flipCode]
                           { return SubmitRealFlip(op, cudaStream, inTensor, outTensor, flipCode); });
}

extern "C" int32_t cvcudaFlipVarShapeSubmit(void *handle, void *stream, void *input, void *output, void *flipCode)
{
    static FlipVarShapeSubmit submit = ResolveLibrary<FlipVarShapeSubmit>("libcvcuda.so.0", "cvcudaFlipVarShapeSubmit");
    return CallThroughGate(g_flipGate, "var-shape flip gate", false,
                           [handle, stream, input, output, flipCode]
                           { return submit(handle, stream, input, output, flipCode); });
}

extern "C" int32_t cvcudaFindHomographySubmit(void *handle, void *stream, void *srcPts, void *dstPts, void *models)
{
    static FindHomographySubmit submit
        = ResolveLibrary<FindHomographySubmit>("libcvcuda.so.0", "cvcudaFindHomographySubmit");
    return SubmitThroughGate(g_findHomographyGate, "find-homography gate", static_cast<cudaStream_t>(stream),
                             [handle, stream, srcPts, dstPts, models]
                             { return submit(handle, stream, srcPts, dstPts, models); });
}

extern "C" int32_t cvcudaMorphologySubmit(void *handle, void *stream, void *input, void *output, void *workspace,
                                          int32_t morphType, int32_t maskWidth, int32_t maskHeight, int32_t anchorX,
                                          int32_t anchorY, int32_t iteration, int32_t borderMode)
{
    static MorphologySubmit submit = ResolveLibrary<MorphologySubmit>("libcvcuda.so.0", "cvcudaMorphologySubmit");
    return SubmitThroughGate(g_morphologyGate, "morphology gate", static_cast<cudaStream_t>(stream),
                             [handle, stream, input, output, workspace, morphType, maskWidth, maskHeight, anchorX,
                              anchorY, iteration, borderMode]
                             {
                                 return submit(handle, stream, input, output, workspace, morphType, maskWidth,
                                               maskHeight, anchorX, anchorY, iteration, borderMode);
                             });
}

extern "C" int32_t cvcudaMorphologyVarShapeSubmit(void *handle, void *stream, void *input, void *output,
                                                  void *workspace, int32_t morphType, void *masks, void *anchors,
                                                  int32_t iteration, int32_t borderMode)
{
    static MorphologyVarShapeSubmit submit
        = ResolveLibrary<MorphologyVarShapeSubmit>("libcvcuda.so.0", "cvcudaMorphologyVarShapeSubmit");
    return SubmitThroughGate(
        g_morphologyGate, "var-shape morphology gate", static_cast<cudaStream_t>(stream),
        [handle, stream, input, output, workspace, morphType, masks, anchors, iteration, borderMode]
        { return submit(handle, stream, input, output, workspace, morphType, masks, anchors, iteration, borderMode); });
}

extern "C" int32_t cvcudaStackSubmit(void *handle, void *stream, void *input, void *output)
{
    static StackSubmit submit = ResolveLibrary<StackSubmit>("libcvcuda.so.0", "cvcudaStackSubmit");
    return SubmitThroughGate(g_concurrentStackGate, "tensor stack gate", static_cast<cudaStream_t>(stream),
                             [handle, stream, input, output]
                             {
                                 {
                                     std::lock_guard lock(g_stackMutex);
                                     g_stackSubmitReached = true;
                                 }
                                 g_stackCondition.notify_all();
                                 ReachEntryGate(g_stackGate, "stack-submit gate");
                                 return submit(handle, stream, input, output);
                             });
}

extern "C" int32_t cvcudaStackVarShapeSubmit(void *handle, void *stream, void *input, void *output)
{
    static StackSubmit submit = ResolveLibrary<StackSubmit>("libcvcuda.so.0", "cvcudaStackVarShapeSubmit");
    return SubmitThroughGate(g_concurrentStackGate, "var-shape stack gate", static_cast<cudaStream_t>(stream),
                             [handle, stream, input, output] { return submit(handle, stream, input, output); });
}

extern "C" int32_t nvcvImageBatchVarShapeGetMaxSize(void *handle, int32_t *maxWidth, int32_t *maxHeight)
{
    static ImageBatchGetMaxSize get
        = ResolveLibrary<ImageBatchGetMaxSize>("libnvcv_types.so.0", "nvcvImageBatchVarShapeGetMaxSize");
    return CallThroughGate(g_imageBatchMetadataGate, "image-batch max-size gate", g_detachFirstImageBatchMetadataWaiter,
                           [handle, maxWidth, maxHeight] { return get(handle, maxWidth, maxHeight); });
}

extern "C" int32_t nvcvImageBatchVarShapeGetUniqueFormat(void *handle, void *format)
{
    static ImageBatchGetUniqueFormat get
        = ResolveLibrary<ImageBatchGetUniqueFormat>("libnvcv_types.so.0", "nvcvImageBatchVarShapeGetUniqueFormat");
    return CallThroughGate(g_imageBatchMetadataGate, "image-batch unique-format gate",
                           g_detachFirstImageBatchMetadataWaiter, [handle, format] { return get(handle, format); });
}

extern "C" void cvcudaTestArmFlipGate()
{
    g_detachFirstFlipWaiter = false;
    ArmGate(g_flipGate);
}

extern "C" void cvcudaTestArmDetachedFlipGate()
{
    g_detachFirstFlipWaiter = true;
    ArmGate(g_flipGate);
}

extern "C" int cvcudaTestWaitForFirstFlip(int timeoutMilliseconds)
{
    return WaitForGateState(g_flipGate, timeoutMilliseconds, &Gate::firstReached) ? 1 : 0;
}

extern "C" int cvcudaTestWaitForSecondFlip(int timeoutMilliseconds)
{
    return WaitForGateState(g_flipGate, timeoutMilliseconds, &Gate::secondReached) ? 1 : 0;
}

extern "C" int cvcudaTestFlipOverlapObserved()
{
    std::lock_guard lock(g_flipGate.mutex);
    return g_flipGate.secondReachedBeforeRelease ? 1 : 0;
}

extern "C" void cvcudaTestReleaseFlip()
{
    ReleaseGate(g_flipGate);
}

extern "C" void cvcudaTestReleaseFlipAndObserveSecond()
{
    ReleaseGate(g_flipGate, true);
}

extern "C" void cvcudaTestArmFindHomographyGate()
{
    ArmGate(g_findHomographyGate);
}

extern "C" int cvcudaTestWaitForFirstFindHomography(int timeoutMilliseconds)
{
    return WaitForGateState(g_findHomographyGate, timeoutMilliseconds, &Gate::firstReached) ? 1 : 0;
}

extern "C" int cvcudaTestWaitForSecondFindHomography(int timeoutMilliseconds)
{
    return WaitForGateState(g_findHomographyGate, timeoutMilliseconds, &Gate::secondReached) ? 1 : 0;
}

extern "C" void cvcudaTestReleaseFindHomographyAndObserveSecond()
{
    ReleaseGate(g_findHomographyGate, true);
}

extern "C" void cvcudaTestArmMorphologyGate()
{
    ArmGate(g_morphologyGate);
}

extern "C" int cvcudaTestWaitForFirstMorphology(int timeoutMilliseconds)
{
    return WaitForGateState(g_morphologyGate, timeoutMilliseconds, &Gate::firstReached) ? 1 : 0;
}

extern "C" int cvcudaTestWaitForSecondMorphology(int timeoutMilliseconds)
{
    return WaitForGateState(g_morphologyGate, timeoutMilliseconds, &Gate::secondReached) ? 1 : 0;
}

extern "C" void cvcudaTestReleaseMorphologyAndObserveSecond()
{
    ReleaseGate(g_morphologyGate, true);
}

extern "C" void cvcudaTestArmConcurrentStackGate()
{
    ArmGate(g_concurrentStackGate);
}

extern "C" int cvcudaTestWaitForFirstConcurrentStack(int timeoutMilliseconds)
{
    return WaitForGateState(g_concurrentStackGate, timeoutMilliseconds, &Gate::firstReached) ? 1 : 0;
}

extern "C" int cvcudaTestWaitForSecondConcurrentStack(int timeoutMilliseconds)
{
    return WaitForGateState(g_concurrentStackGate, timeoutMilliseconds, &Gate::secondReached) ? 1 : 0;
}

extern "C" void cvcudaTestReleaseConcurrentStack()
{
    ReleaseGate(g_concurrentStackGate);
}

extern "C" void cvcudaTestReleaseConcurrentStackAndObserveSecond()
{
    ReleaseGate(g_concurrentStackGate, true);
}

extern "C" void cvcudaTestArmImageBatchMetadataGate(int detachFirstWaiter)
{
    g_detachFirstImageBatchMetadataWaiter = detachFirstWaiter != 0;
    ArmGate(g_imageBatchMetadataGate);
}

extern "C" int cvcudaTestWaitForFirstImageBatchMetadata(int timeoutMilliseconds)
{
    return WaitForGateState(g_imageBatchMetadataGate, timeoutMilliseconds, &Gate::firstReached) ? 1 : 0;
}

extern "C" int cvcudaTestWaitForSecondImageBatchMetadata(int timeoutMilliseconds)
{
    return WaitForGateState(g_imageBatchMetadataGate, timeoutMilliseconds, &Gate::secondReached) ? 1 : 0;
}

extern "C" void cvcudaTestReleaseImageBatchMetadata()
{
    ReleaseGate(g_imageBatchMetadataGate);
}

extern "C" void cvcudaTestReleaseImageBatchMetadataAndObserveSecond()
{
    ReleaseGate(g_imageBatchMetadataGate, true);
}

extern "C" void cvcudaTestArmFlipReturnProbe()
{
    ArmEntryGate(g_flipReturnGate, 1);
}

extern "C" int cvcudaTestWaitForFlipReturn(int timeoutMilliseconds)
{
    return WaitForEntryGate(g_flipReturnGate, timeoutMilliseconds);
}

extern "C" void cvcudaTestArmLifetimeGate()
{
    std::lock_guard lock(g_lifetimeMutex);
    if (g_lifetimeGateEvent == nullptr
        && cudaEventCreateWithFlags(&g_lifetimeGateEvent, cudaEventDisableTiming) != cudaSuccess)
    {
        std::abort();
    }
    g_lifetimeArmed                  = true;
    g_lifetimeReached                = false;
    g_lifetimeRelease                = false;
    g_lifetimeSubmitReturned         = false;
    g_lifetimeActive                 = true;
    g_destroyedBeforeLifetimeRelease = false;
    g_resourceTracking               = true;
    g_trackOutputResources           = true;
    g_trackedTensors.clear();
    g_destroyedTensors.clear();
}

extern "C" int cvcudaTestWaitForLifetimeGate(int timeoutMilliseconds)
{
    std::unique_lock lock(g_lifetimeMutex);
    bool             reached = g_lifetimeCondition.wait_for(lock, std::chrono::milliseconds(timeoutMilliseconds),
                                                            [] { return g_lifetimeReached; });
    return reached ? 1 : 0;
}

extern "C" int cvcudaTestWaitForLifetimeSubmit(int timeoutMilliseconds)
{
    std::unique_lock lock(g_lifetimeMutex);
    bool             returned = g_lifetimeCondition.wait_for(lock, std::chrono::milliseconds(timeoutMilliseconds),
                                                             [] { return g_lifetimeSubmitReturned; });
    return returned ? 1 : 0;
}

extern "C" int cvcudaTestLifetimeWorkPending()
{
    cudaError_t status = cudaEventQuery(g_lifetimeGateEvent);
    if (status == cudaErrorNotReady)
    {
        return 1;
    }
    if (status == cudaSuccess)
    {
        return 0;
    }
    std::abort();
}

extern "C" void cvcudaTestReleaseLifetimeGate()
{
    {
        std::lock_guard lock(g_lifetimeMutex);
        g_lifetimeRelease = true;
    }
    g_lifetimeCondition.notify_all();
}

extern "C" int cvcudaTestTensorDestroyedBeforeLifetimeRelease()
{
    std::lock_guard lock(g_lifetimeMutex);
    return g_destroyedBeforeLifetimeRelease ? 1 : 0;
}

extern "C" void cvcudaTestStopResourceDestructionTracking()
{
    std::lock_guard lock(g_lifetimeMutex);
    g_resourceTracking = false;
}

extern "C" int cvcudaTestTrackedTensorCount()
{
    std::lock_guard lock(g_lifetimeMutex);
    return static_cast<int>(g_trackedTensors.size());
}

extern "C" int cvcudaTestDestroyedTensorCount()
{
    std::lock_guard lock(g_lifetimeMutex);
    return static_cast<int>(g_destroyedTensors.size());
}

extern "C" void cvcudaTestArmNextBusyMutexAttempt()
{
    RealPthreadMutexTryLock();
    g_busyMutexThread.store(pthread_self(), std::memory_order_relaxed);
    ArmEntryGate(g_busyMutexAttemptGate, 1);
    g_probeNextBusyMutexAttempt.store(true, std::memory_order_release);
}

extern "C" int cvcudaTestWaitForBusyMutexAttempt(int timeoutMilliseconds)
{
    return WaitForEntryGate(g_busyMutexAttemptGate, timeoutMilliseconds);
}

extern "C" void cvcudaTestReleaseBusyMutexAttempt()
{
    g_probeNextBusyMutexAttempt.store(false, std::memory_order_release);
    g_busyMutexThread.store(0, std::memory_order_relaxed);
    ReleaseEntryGate(g_busyMutexAttemptGate);
}

extern "C" void cvcudaTestArmDelayedFlipRelease()
{
    std::lock_guard lock(g_delayedFlipReleaseMutex);
    g_delayedFlipReleaseStarted = false;
    g_delayedFlipReleaseReached = false;
}

extern "C" int cvcudaTestWaitForDelayedFlipRelease(int timeoutMilliseconds)
{
    std::unique_lock lock(g_delayedFlipReleaseMutex);
    bool started = g_delayedFlipReleaseCondition.wait_for(lock, std::chrono::milliseconds(timeoutMilliseconds),
                                                          [] { return g_delayedFlipReleaseStarted; });
    return started ? 1 : 0;
}

extern "C" int cvcudaTestDelayedFlipReleaseReached()
{
    std::lock_guard lock(g_delayedFlipReleaseMutex);
    return g_delayedFlipReleaseReached ? 1 : 0;
}

extern "C" void cvcudaTestReleaseFlipAfterDelay(int delayMilliseconds)
{
    {
        std::lock_guard lock(g_delayedFlipReleaseMutex);
        g_delayedFlipReleaseStarted = true;
    }
    g_delayedFlipReleaseCondition.notify_all();

    std::this_thread::sleep_for(std::chrono::milliseconds(delayMilliseconds));
    {
        std::lock_guard lock(g_delayedFlipReleaseMutex);
        g_delayedFlipReleaseReached = true;
    }
    g_delayedFlipReleaseCondition.notify_all();
    ReleaseGate(g_flipGate);
}

extern "C" void cvcudaTestArmReaderEntryGate(int targetCalls)
{
    ArmEntryGate(g_readerEntryGate, targetCalls);
}

extern "C" void cvcudaTestReachReaderEntry()
{
    ReachEntryGate(g_readerEntryGate, "reader-entry gate");
}

extern "C" int cvcudaTestWaitForReaderEntry(int timeoutMilliseconds)
{
    return WaitForEntryGate(g_readerEntryGate, timeoutMilliseconds);
}

extern "C" void cvcudaTestReleaseReaderEntryGate()
{
    ReleaseEntryGate(g_readerEntryGate);
}

extern "C" void cvcudaTestArmBatchMutation(int action)
{
    {
        std::lock_guard lock(g_batchMutex);
        g_armedBatchAction  = action;
        g_batchMutationSeen = false;
        g_releaseBatch      = false;
    }
    {
        std::lock_guard lock(g_stackMutex);
        g_stackSubmitReached = false;
    }
}

extern "C" int cvcudaTestWaitForBatchPush(int timeoutMilliseconds)
{
    std::unique_lock lock(g_batchMutex);
    bool             reached = g_batchCondition.wait_for(lock, std::chrono::milliseconds(timeoutMilliseconds),
                                                         [] { return g_batchMutationSeen; });
    return reached ? 1 : 0;
}

extern "C" int cvcudaTestWaitForStackSubmit(int timeoutMilliseconds)
{
    std::unique_lock lock(g_stackMutex);
    bool             reached = g_stackCondition.wait_for(lock, std::chrono::milliseconds(timeoutMilliseconds),
                                                         [] { return g_stackSubmitReached; });
    return reached ? 1 : 0;
}

extern "C" void cvcudaTestReleaseBatchPush()
{
    {
        std::lock_guard lock(g_batchMutex);
        g_releaseBatch = true;
    }
    g_batchCondition.notify_all();
}

extern "C" void cvcudaTestArmStackGate()
{
    {
        std::lock_guard lock(g_stackMutex);
        g_stackSubmitReached = false;
    }
    ArmEntryGate(g_stackGate, 1);
}

extern "C" void cvcudaTestReleaseStackGate()
{
    ReleaseEntryGate(g_stackGate);
}

extern "C" void cvcudaTestArmGpuGate()
{
    ArmEntryGate(g_gpuGate, 1);
}

extern "C" int cvcudaTestQueueGpuGate(void *stream)
{
    cudaError_t status = cudaLaunchHostFunc(reinterpret_cast<cudaStream_t>(stream), ReachGpuGate, nullptr);
    return status == cudaSuccess ? 1 : 0;
}

extern "C" int cvcudaTestWaitForGpuGate(int timeoutMilliseconds)
{
    return WaitForEntryGate(g_gpuGate, timeoutMilliseconds);
}

extern "C" void cvcudaTestReleaseGpuGate()
{
    ReleaseEntryGate(g_gpuGate);
}

extern "C" void cvcudaTestArmEventRecordGate()
{
    EnableEventRecordProbe();
    g_eventRecordAtEntry.store(false);
    ArmEntryGate(g_eventRecordGate, 1);
}

extern "C" void cvcudaTestArmEventRecordEntryGate()
{
    EnableEventRecordProbe();
    g_eventRecordAtEntry.store(true);
    ArmEntryGate(g_eventRecordGate, 1);
}

extern "C" int cvcudaTestWaitForEventRecordCount(int targetCalls, int timeoutMilliseconds)
{
    std::unique_lock lock(g_eventRecordGate.mutex);
    bool             reached
        = g_eventRecordGate.condition.wait_for(lock, std::chrono::milliseconds(timeoutMilliseconds),
                                               [targetCalls] { return g_eventRecordGate.calls >= targetCalls; });
    return reached ? 1 : 0;
}

extern "C" void cvcudaTestReleaseEventRecordGate()
{
    ReleaseEntryGate(g_eventRecordGate);
}

extern "C" void cvcudaTestArmDeviceSynchronizeGate()
{
    EnableDeviceSynchronizeProbe();
    ArmEntryGate(g_deviceSynchronizeGate, 1);
}

extern "C" int cvcudaTestWaitForDeviceSynchronize(int timeoutMilliseconds)
{
    return WaitForEntryGate(g_deviceSynchronizeGate, timeoutMilliseconds);
}

extern "C" int cvcudaTestWaitForDeviceSynchronizeCalls(int calls, int timeoutMilliseconds)
{
    std::unique_lock lock(g_deviceSynchronizeGate.mutex);
    return g_deviceSynchronizeGate.condition.wait_for(
        lock, std::chrono::milliseconds(timeoutMilliseconds),
        [calls] { return g_deviceSynchronizeGate.armed && g_deviceSynchronizeGate.calls >= calls; });
}

extern "C" void cvcudaTestReleaseDeviceSynchronizeGate()
{
    ReleaseEntryGate(g_deviceSynchronizeGate);
}

extern "C" void cvcudaTestArmGpuSignalGate()
{
    std::lock_guard lock(g_gpuSignalMutex);
    if (g_gpuSignalEvent != nullptr)
    {
        cudaEventDestroy(g_gpuSignalEvent);
        g_gpuSignalEvent = nullptr;
    }
    if (cudaEventCreateWithFlags(&g_gpuSignalEvent, cudaEventDisableTiming) != cudaSuccess)
    {
        std::abort();
    }
}

extern "C" int cvcudaTestQueueGpuSignal(void *stream)
{
    std::lock_guard lock(g_gpuSignalMutex);
    if (g_gpuSignalEvent == nullptr)
    {
        return 0;
    }
    cudaError_t status = cudaEventRecord(g_gpuSignalEvent, reinterpret_cast<cudaStream_t>(stream));
    return status == cudaSuccess ? 1 : 0;
}

extern "C" int cvcudaTestWaitForGpuSignal(int timeoutMilliseconds)
{
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMilliseconds);
    while (std::chrono::steady_clock::now() < deadline)
    {
        cudaError_t status;
        {
            std::lock_guard lock(g_gpuSignalMutex);
            if (g_gpuSignalEvent == nullptr)
            {
                return 0;
            }
            status = cudaEventQuery(g_gpuSignalEvent);
        }
        if (status == cudaSuccess)
        {
            return 1;
        }
        if (status != cudaErrorNotReady)
        {
            return 0;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return 0;
}

extern "C" void cvcudaTestReleaseGpuSignalGate()
{
    std::lock_guard lock(g_gpuSignalMutex);
    if (g_gpuSignalEvent != nullptr)
    {
        cudaEventDestroy(g_gpuSignalEvent);
        g_gpuSignalEvent = nullptr;
    }
}

extern "C" void cvcudaTestArmImageCpuCopyFailure()
{
    EnableImageCpuCopyProbe();
    std::lock_guard lock(g_imageCpuCopyGate.mutex);
    g_imageCpuCopyGate.armed              = true;
    g_imageCpuCopyGate.firstReached       = false;
    g_imageCpuCopyGate.secondReached      = false;
    g_imageCpuCopyGate.cleanupSyncReached = false;
    g_imageCpuCopyGate.released           = false;
    g_imageCpuCopyGate.pending            = false;
    g_imageCpuCopyGate.calls              = 0;
    g_imageCpuCopyGate.api                = 0;
    g_imageCpuCopyGate.copyFailureThread  = {};
}

extern "C" int cvcudaTestWaitForImageCpuFirstCopy(int timeoutMilliseconds)
{
    std::unique_lock lock(g_imageCpuCopyGate.mutex);
    bool reached = g_imageCpuCopyGate.condition.wait_for(lock, std::chrono::milliseconds(timeoutMilliseconds),
                                                         [] { return g_imageCpuCopyGate.firstReached; });
    return reached ? 1 : 0;
}

extern "C" int cvcudaTestWaitForImageCpuCopyFailure(int timeoutMilliseconds)
{
    std::unique_lock lock(g_imageCpuCopyGate.mutex);
    bool reached = g_imageCpuCopyGate.condition.wait_for(lock, std::chrono::milliseconds(timeoutMilliseconds),
                                                         [] { return g_imageCpuCopyGate.secondReached; });
    return reached ? 1 : 0;
}

extern "C" int cvcudaTestWaitForImageCpuCleanupSync(int timeoutMilliseconds)
{
    std::unique_lock lock(g_imageCpuCopyGate.mutex);
    bool reached = g_imageCpuCopyGate.condition.wait_for(lock, std::chrono::milliseconds(timeoutMilliseconds),
                                                         [] { return g_imageCpuCopyGate.cleanupSyncReached; });
    return reached ? 1 : 0;
}

extern "C" int cvcudaTestImageCpuCopyPending()
{
    std::lock_guard lock(g_imageCpuCopyGate.mutex);
    return g_imageCpuCopyGate.pending ? 1 : 0;
}

extern "C" int cvcudaTestImageCpuCopyApi()
{
    std::lock_guard lock(g_imageCpuCopyGate.mutex);
    return g_imageCpuCopyGate.api;
}

extern "C" void cvcudaTestReleaseImageCpuFirstCopy()
{
    {
        std::lock_guard lock(g_imageCpuCopyGate.mutex);
        g_imageCpuCopyGate.released = true;
        if (g_imageCpuCopyGate.api == 2)
        {
            g_imageCpuCopyGate.pending = false;
        }
        g_imageCpuCopyGate.armed = false;
    }
    g_imageCpuCopyGate.condition.notify_all();
}

extern "C" void cvcudaTestArmCriticalSectionHold()
{
    ArmEntryGate(g_criticalSectionHoldGate, 1);
}

extern "C" int cvcudaTestWaitForCriticalSectionHold(int timeoutMilliseconds)
{
    return WaitForEntryGate(g_criticalSectionHoldGate, timeoutMilliseconds);
}

extern "C" void cvcudaTestHoldCriticalSection(void *targetObject)
{
    // PyCriticalSection is two pointer-sized fields in the supported
    // free-threaded CPython ABIs. Keep extra aligned storage so this test probe
    // remains insulated from small private-layout changes between releases.
    alignas(std::max_align_t) std::array<std::uintptr_t, 8> criticalSection{};
    RealCriticalSectionBegin()(criticalSection.data(), targetObject);
    ReachEntryGate(g_criticalSectionHoldGate, "critical-section gate");
    RealCriticalSectionEnd()(criticalSection.data());
}

extern "C" void cvcudaTestReleaseCriticalSectionHold()
{
    ReleaseEntryGate(g_criticalSectionHoldGate);
}

extern "C" void *cvcudaTestTakeDLPackManager(void *capsule, int versioned)
{
    const char *name     = versioned != 0 ? "dltensor_versioned" : "dltensor";
    const char *usedName = versioned != 0 ? "used_dltensor_versioned" : "used_dltensor";
    void       *tensor   = PyCapsule_GetPointer(capsule, name);
    if (tensor == nullptr || PyCapsule_SetName(capsule, usedName) != 0)
    {
        return nullptr;
    }
    return tensor;
}

static void DeleteDLPackManager(DLPackManager *managedTensor, int versioned)
{
    if (versioned != 0)
    {
        auto *tensor = reinterpret_cast<DLManagedTensorVersioned *>(managedTensor);
        if (tensor->deleter == nullptr)
        {
            std::abort();
        }
        tensor->deleter(tensor);
    }
    else
    {
        auto *tensor = reinterpret_cast<DLManagedTensor *>(managedTensor);
        if (tensor->deleter == nullptr)
        {
            std::abort();
        }
        tensor->deleter(tensor);
    }
}

extern "C" int cvcudaTestDeleteDLPackManagerOnNativeThread(void *managedTensor, int versioned)
{
    if (managedTensor == nullptr)
    {
        return 0;
    }

    auto            *manager   = static_cast<DLPackManager *>(managedTensor);
    std::atomic_bool completed = false;
    std::thread      deleter( // NOSONAR: this C++17 target cannot use std::jthread.
        [manager, versioned, &completed]
        {
            g_nativeDLPackThreadAttached.store(PyGILState_Check(), std::memory_order_release); // NOSONAR: test handoff.
            DeleteDLPackManager(manager, versioned);
            completed.store(true, std::memory_order_release); // NOSONAR: test handoff.
        });
    deleter.join();
    return completed.load(std::memory_order_acquire) ? 1 : 0; // NOSONAR: test handoff.
}

extern "C" int cvcudaTestNativeDLPackDeleteThreadWasAttached()
{
    return g_nativeDLPackThreadAttached.load(std::memory_order_acquire); // NOSONAR: test handoff.
}

extern "C" int32_t nvcvTensorBatchPushTensors(void *handle, void *const *tensors, int32_t numTensors)
{
    static BatchPush push = ResolveLibrary<BatchPush>("libnvcv_types.so.0", "nvcvTensorBatchPushTensors");
    return PauseBatchMutation(0, push(handle, tensors, numTensors));
}

extern "C" int32_t nvcvImageBatchVarShapePushImages(void *handle, void *const *images, int32_t numImages)
{
    static BatchPush push = ResolveLibrary<BatchPush>("libnvcv_types.so.0", "nvcvImageBatchVarShapePushImages");
    return PauseBatchMutation(1, push(handle, images, numImages));
}

extern "C" int32_t nvcvTensorBatchPopTensors(void *handle, int32_t numTensors)
{
    static BatchPop pop = ResolveLibrary<BatchPop>("libnvcv_types.so.0", "nvcvTensorBatchPopTensors");
    return PauseBatchMutation(2, pop(handle, numTensors));
}

extern "C" int32_t nvcvTensorBatchClear(void *handle)
{
    static BatchClear clear = ResolveLibrary<BatchClear>("libnvcv_types.so.0", "nvcvTensorBatchClear");
    return PauseBatchMutation(3, clear(handle));
}

extern "C" int32_t nvcvTensorBatchSetTensors(void *handle, int32_t index, void *const *tensors, int32_t numTensors)
{
    static BatchSet set = ResolveLibrary<BatchSet>("libnvcv_types.so.0", "nvcvTensorBatchSetTensors");
    return PauseBatchMutation(4, set(handle, index, tensors, numTensors));
}

extern "C" int32_t nvcvImageBatchVarShapePopImages(void *handle, int32_t numImages)
{
    static BatchPop pop = ResolveLibrary<BatchPop>("libnvcv_types.so.0", "nvcvImageBatchVarShapePopImages");
    return PauseBatchMutation(5, pop(handle, numImages));
}

extern "C" int32_t nvcvImageBatchVarShapeClear(void *handle)
{
    static BatchClear clear = ResolveLibrary<BatchClear>("libnvcv_types.so.0", "nvcvImageBatchVarShapeClear");
    return PauseBatchMutation(6, clear(handle));
}

extern "C" NVCVStatus nvcvImageExportData(NVCVImageHandle handle, NVCVImageData *data)
{
    static ImageExportData exportData = ResolveLibrary<ImageExportData>("libnvcv_types.so.0", "nvcvImageExportData");
    NVCVStatus             status     = exportData(handle, data);
    {
        std::lock_guard lock(g_imageCpuCopyGate.mutex);
        if (g_imageCpuCopyGate.armed && status == NVCV_SUCCESS && data != nullptr
            && data->bufferType == NVCV_IMAGE_BUFFER_STRIDED_CUDA && data->buffer.strided.numPlanes >= 2)
        {
            data->buffer.strided.planes[1].basePtr = nullptr;
        }
    }
    return status;
}

extern "C" NVCVStatus nvcvTensorDecRef(NVCVTensorHandle handle, int *newRefCount)
{
    static TensorRefCount getRefCount = ResolveLibrary<TensorRefCount>("libnvcv_types.so.0", "nvcvTensorRefCount");
    static TensorDecRef   decRef      = ResolveLibrary<TensorDecRef>("libnvcv_types.so.0", "nvcvTensorDecRef");
    bool                  tracked;
    {
        std::lock_guard lock(g_lifetimeMutex);
        tracked = g_trackedTensors.contains(handle);
    }
    if (tracked)
    {
        int refCount = -1;
        if (getRefCount(handle, &refCount) == NVCV_SUCCESS && refCount == 1)
        {
            // Record the final release before calling through: device-memory
            // teardown may synchronize with work that the lifetime test has
            // deliberately gated, but ownership ended at this call boundary.
            std::lock_guard lock(g_lifetimeMutex);
            g_destroyedTensors.insert(handle);
            if (g_lifetimeActive && !g_lifetimeRelease)
            {
                g_destroyedBeforeLifetimeRelease = true;
            }
        }
    }

    int        observedRefCount = -1;
    NVCVStatus status           = decRef(handle, &observedRefCount);
    if (newRefCount != nullptr)
    {
        *newRefCount = observedRefCount;
    }
    if (status == NVCV_SUCCESS && observedRefCount == 0)
    {
        std::lock_guard lock(g_lifetimeMutex);
        if (g_trackedTensors.contains(handle))
        {
            g_destroyedTensors.insert(handle);
            if (g_lifetimeActive && !g_lifetimeRelease)
            {
                g_destroyedBeforeLifetimeRelease = true;
            }
        }
    }
    return status;
}
