//! Who a process is (its SID) and the security descriptors that keep a directory, a pipe and an
//! event to that one user.

use std::path::Path;

use windows_sys::Win32::Foundation::{LocalFree, HANDLE};
use windows_sys::Win32::Security::Authorization::{
    ConvertSidToStringSidW, ConvertStringSecurityDescriptorToSecurityDescriptorW, GetNamedSecurityInfoW, SetNamedSecurityInfoW, SDDL_REVISION_1, SE_FILE_OBJECT,
};
use windows_sys::Win32::Security::{
    GetAce, GetAclInformation, GetSecurityDescriptorControl, GetSecurityDescriptorDacl, GetTokenInformation, AclSizeInformation, ACCESS_ALLOWED_ACE, ACL, ACL_SIZE_INFORMATION,
    DACL_SECURITY_INFORMATION, PROTECTED_DACL_SECURITY_INFORMATION, PSECURITY_DESCRIPTOR, SECURITY_ATTRIBUTES, SE_DACL_PROTECTED, TokenUser, TOKEN_QUERY, TOKEN_USER,
};
use windows_sys::Win32::System::Threading::{GetCurrentProcess, OpenProcess, OpenProcessToken, PROCESS_QUERY_LIMITED_INFORMATION};

use super::{wide, Handle};
use crate::error::Result;
use crate::sys::Principal;

fn failed() -> std::io::Error {
    std::io::Error::last_os_error()
}

/// A security descriptor the system allocated, freed when dropped.
pub(crate) struct Descriptor(PSECURITY_DESCRIPTOR);

// SAFETY: the descriptor is plain memory nothing mutates after creation.
unsafe impl Send for Descriptor {}
// SAFETY: as above, it is only read.
unsafe impl Sync for Descriptor {}

impl Descriptor {
    /// Parse ``sddl``.
    fn parse(sddl: &str) -> std::io::Result<Descriptor> {
        let text = wide(sddl);
        let mut out: PSECURITY_DESCRIPTOR = std::ptr::null_mut();
        // SAFETY: `text` is NUL-ended; `out` receives a LocalAlloc'd descriptor.
        if unsafe { ConvertStringSecurityDescriptorToSecurityDescriptorW(text.as_ptr(), SDDL_REVISION_1, &mut out, std::ptr::null_mut()) } == 0 {
            return Err(failed());
        }
        Ok(Descriptor(out))
    }

    /// The attributes that give a new kernel object this descriptor and keep its handle out of children.
    pub(crate) fn attributes(&self) -> SECURITY_ATTRIBUTES {
        SECURITY_ATTRIBUTES { nLength: std::mem::size_of::<SECURITY_ATTRIBUTES>() as u32, lpSecurityDescriptor: self.0, bInheritHandle: 0 }
    }
}

impl Drop for Descriptor {
    fn drop(&mut self) {
        // SAFETY: allocated by the system with LocalAlloc and freed once.
        unsafe { LocalFree(self.0) };
    }
}

/// A descriptor that grants ``access`` (an SDDL right) to the current user and to no one else.
pub(crate) fn only_me(access: &str, inherit: &str) -> Result<Descriptor> {
    let me = current_principal()?;
    Ok(Descriptor::parse(&format!("D:P(A;{inherit};{access};;;{})", me.0))?)
}

/// The text form of a SID the system owns.
fn sid_text(sid: *mut core::ffi::c_void) -> std::io::Result<String> {
    let mut out: *mut u16 = std::ptr::null_mut();
    // SAFETY: `sid` is a valid SID for the call; `out` receives a LocalAlloc'd string.
    if unsafe { ConvertSidToStringSidW(sid, &mut out) } == 0 {
        return Err(failed());
    }
    // SAFETY: `out` is NUL-ended.
    let len = (0..).take_while(|i| unsafe { *out.add(*i) } != 0).count();
    // SAFETY: `len` units are initialised.
    let text = String::from_utf16_lossy(unsafe { std::slice::from_raw_parts(out, len) });
    // SAFETY: allocated by the system with LocalAlloc and freed once.
    unsafe { LocalFree(out.cast()) };
    Ok(text)
}

/// The user of an access token, as a SID string.
pub(crate) fn token_sid(token: HANDLE) -> std::io::Result<String> {
    let mut need = 0u32;
    // SAFETY: the first call only asks how many bytes TokenUser needs.
    unsafe { GetTokenInformation(token, TokenUser, std::ptr::null_mut(), 0, &mut need) };
    let mut buf = vec![0u64; (need as usize).div_ceil(8).max(1)];
    // SAFETY: `buf` holds `need` bytes, aligned for TOKEN_USER.
    if unsafe { GetTokenInformation(token, TokenUser, buf.as_mut_ptr().cast(), need, &mut need) } == 0 {
        return Err(failed());
    }
    // SAFETY: the call filled a TOKEN_USER at the start of `buf`.
    sid_text(unsafe { (*buf.as_ptr().cast::<TOKEN_USER>()).User.Sid })
}

fn sid_of_process_handle(process: HANDLE) -> std::io::Result<String> {
    let mut token: HANDLE = std::ptr::null_mut();
    // SAFETY: `process` is open; `token` receives a handle we own.
    if unsafe { OpenProcessToken(process, TOKEN_QUERY, &mut token) } == 0 {
        return Err(failed());
    }
    let token = Handle(token);
    token_sid(token.0)
}

pub(in crate::sys) fn current_principal() -> Result<Principal> {
    // SAFETY: the pseudo handle of this process needs no closing.
    Ok(Principal::new(sid_of_process_handle(unsafe { GetCurrentProcess() })?))
}

/// The user another process runs as; an error when it cannot be opened (another user's can not).
pub(crate) fn process_principal(pid: u32) -> std::io::Result<String> {
    // SAFETY: opens a handle we own.
    let process = Handle::checked(unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid) })?;
    sid_of_process_handle(process.0)
}

/// Create ``path`` and its parents; ``path`` itself is then left to the current user alone, and what is made inside inherits that.
pub fn private_dir(path: &Path) -> Result<()> {
    std::fs::create_dir_all(path)?;
    let only = only_me("FA", "OICI")?;
    let (mut present, mut defaulted) = (0i32, 0i32);
    let mut dacl: *mut ACL = std::ptr::null_mut();
    // SAFETY: the descriptor is valid; the out-pointers are valid for the call.
    if unsafe { GetSecurityDescriptorDacl(only.0, &mut present, &mut dacl, &mut defaulted) } == 0 {
        return Err(failed().into());
    }
    let name = wide(path);
    let kind = DACL_SECURITY_INFORMATION | PROTECTED_DACL_SECURITY_INFORMATION;
    // SAFETY: `name` is NUL-ended and `dacl` points into `only`, which outlives the call.
    let rc = unsafe { SetNamedSecurityInfoW(name.as_ptr().cast_mut(), SE_FILE_OBJECT, kind, std::ptr::null_mut(), std::ptr::null_mut(), dacl, std::ptr::null_mut()) };
    if rc != 0 {
        return Err(std::io::Error::from_raw_os_error(rc as i32).into());
    }
    Ok(())
}

/// Whether ``path`` has a DACL that does not inherit and allows exactly the current user.
pub fn owner_only(path: &Path) -> bool {
    owner_only_check(path).unwrap_or(false)
}

fn owner_only_check(path: &Path) -> Result<bool> {
    let name = wide(path);
    let (mut dacl, mut descriptor): (*mut ACL, PSECURITY_DESCRIPTOR) = (std::ptr::null_mut(), std::ptr::null_mut());
    // SAFETY: `name` is NUL-ended; the out-pointers are valid for the call.
    let rc = unsafe { GetNamedSecurityInfoW(name.as_ptr(), SE_FILE_OBJECT, DACL_SECURITY_INFORMATION, std::ptr::null_mut(), std::ptr::null_mut(), &mut dacl, std::ptr::null_mut(), &mut descriptor) };
    if rc != 0 {
        return Err(std::io::Error::from_raw_os_error(rc as i32).into());
    }
    let held = Descriptor(descriptor);
    let (mut control, mut revision) = (0u16, 0u32);
    // SAFETY: the descriptor is valid; the out-pointers are valid for the call.
    if unsafe { GetSecurityDescriptorControl(held.0, &mut control, &mut revision) } == 0 || control & SE_DACL_PROTECTED == 0 {
        return Ok(false);
    }
    // SAFETY: an all-zero ACL_SIZE_INFORMATION is valid; the call fills it.
    let mut size: ACL_SIZE_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: `dacl` is valid inside `held` and `size` is the struct this class asks for.
    if unsafe { GetAclInformation(dacl, (&mut size as *mut ACL_SIZE_INFORMATION).cast(), std::mem::size_of::<ACL_SIZE_INFORMATION>() as u32, AclSizeInformation) } == 0 || size.AceCount != 1 {
        return Ok(false);
    }
    let mut ace: *mut core::ffi::c_void = std::ptr::null_mut();
    // SAFETY: the DACL has one entry and `ace` receives a pointer into it.
    if unsafe { GetAce(dacl, 0, &mut ace) } == 0 {
        return Ok(false);
    }
    let sid = unsafe { std::ptr::addr_of_mut!((*ace.cast::<ACCESS_ALLOWED_ACE>()).SidStart) };
    Ok(sid_text(sid.cast())? == current_principal()?.0)
}
