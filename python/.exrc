if &cp | set nocp | endif
let s:cpo_save=&cpo
set cpo&vim
map! <C-F> gUiw`]a
map Q gq
xmap gx <Plug>NetrwBrowseXVis
nmap gx <Plug>NetrwBrowseX
xnoremap <silent> <Plug>NetrwBrowseXVis :call netrw#BrowseXVis()
nnoremap <silent> <Plug>NetrwBrowseX :call netrw#BrowseX(netrw#GX(),netrw#CheckIfRemote(netrw#GX()))
noremap <Right> <Nop>
noremap <Left> <Nop>
noremap <Down> <Nop>
noremap <Up> <Nop>
map!  gUiw`]a
let &cpo=s:cpo_save
unlet s:cpo_save
set autoindent
set autowrite
set background=dark
set backspace=2
set cindent
set cinoptions=:0,l1,g0,t0,(0,Ws
set expandtab
set exrc
set fileencodings=ucs-bom,utf-8,latin1
set helplang=en
set hlsearch
set incsearch
set makeprg=ninja
set matchpairs=(:),{:},[:],<:>
set ruler
set shiftwidth=4
set showcmd
set softtabstop=4
set suffixes=.bak,~,.o,.h,.info,.swp,.obj,.info,.aux,.log,.dvi,.bbl,.out,.o,.lo
set tags=~/.tags
set viminfo='10,\"100,:20,%,n~/.viminfo
" vim: set ft=vim :
