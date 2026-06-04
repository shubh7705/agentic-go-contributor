## 1. Root Cause
In `Context.SaveUploadedFile` (`context.go`), an unconditional `os.Chmod` call was added after `os.MkdirAll` to support configurable directory permissions. Because `os.MkdirAll` returns nil when a directory already exists, the subsequent `os.Chmod` always executes—even on pre-existing system directories such as `/tmp` that are owned by another user—causing a permission denied error.

## 2. Files to Modify
- `context.go` — Modify the `SaveUploadedFile` method to skip the `os.Chmod` call when the destination directory already exists before the method is invoked.
- `context_test.go` — Add regression tests that verify saving to a pre-existing directory does not alter its permissions and does not error, while ensuring newly created directories still receive the intended permissions.

## 3. New Files to Create
None

## 4. Tests to Update
- `context_test.go` — Add two new test cases under a new test function (e.g., `TestContextSaveUploadedFilePermissions`):
  1. **Pre-existing directory**: Create a temporary directory with a specific permission mode (e.g., `0755`), invoke `SaveUploadedFile` targeting a file inside it, assert no error, assert the directory’s permissions remain unchanged, and assert the file was written.
  2. **Newly created directory**: Invoke `SaveUploadedFile` targeting a file inside a non-existent subdirectory, assert no error, and assert the newly created leaf directory has the exact intended permissions (e.g., `0750`) because `os.Chmod` was applied.

## 5. Implementation Steps
1. Open `context.go` and locate the `SaveUploadedFile` method.
2. Identify the variable holding the target directory path (e.g., `dir := filepath.Dir(dst)`) and the directory permission value used by the current code.
3. **Before** the existing `os.MkdirAll` call, add an existence check using `os.Stat(dir)`:
   - If `os.Stat` returns `nil`, the directory already exists. Set a boolean flag (e.g., `needChmod = false`) and proceed.
   - If `os.Stat` returns `os.IsNotExist(err)`, set `needChmod = true`.
   - If `os.Stat` returns any other error, return that error immediately.
4. Keep the `os.MkdirAll(dir, perm)` call unconditional (or call it when needed) so that intermediate directories are still created and non-directory paths still produce the correct error.
5. Wrap the existing `os.Chmod(dir, perm)` call so it only executes when `needChmod == true`.
6. Leave the multipart file opening (`file.Open()`), destination file creation (`os.Create(dst)`), and `io.Copy` logic untouched.
7. Open `context_test.go` and add the regression test function described in section 4, using `t.TempDir()` for sandboxed directories and `os.Stat` to read back file modes for assertions.

## 6. Risks
- **TOCTOU race condition**: Another process could create the directory between the `os.Stat` and `os.MkdirAll` calls. In that narrow window the method would still execute `os.Chmod` on a directory it did not create. This is acceptable because the acceptance criteria focuses on directories that already exist *before* the call, and skipping `Chmod` on a brand-new directory is far less common than the `/tmp` failure.
- **Behavior change for existing files as leaf path**: If `filepath.Dir(dst)` exists as a regular file (not a directory), the old code would have `os.MkdirAll` return an error. With a pre-check gated only on `os.IsNotExist`, the method may now defer the error until `os.Create(dst)`. To mitigate, the implementation should still invoke `os.MkdirAll` regardless of the `Stat` outcome, only gating `Chmod` on the pre-existence flag.
- **umask interaction**: When `Chmod` is skipped on an existing directory, the permissions of that directory remain as-is. This is the desired behavior, but developers relying on `SaveUploadedFile` to “fix” permissions of an existing directory will no longer get that side effect.

## 7. Validation Strategy
- Run the new regression tests:
  ```bash
  go test -v -run TestContextSaveUploadedFilePermissions ./...
  ```
- Run the full package test suite to ensure no collateral damage:
  ```bash
  go test ./...
  ```
- **Manual /tmp verification** (Linux/macOS): In a small standalone program or test, call `SaveUploadedFile` with a destination path directly inside `/tmp` (e.g., `/tmp/gin_test_<random>.txt`) and confirm it returns `nil` instead of `chmod /tmp: operation not permitted`.
- **Edge cases to check**:
  - Destination is a pre-existing system directory (`/tmp`).
  - Destination is a nested path where the leaf directory does not exist (verifying `Chmod` still runs and sets exact permissions).
  - Destination is a path where the parent is an existing file (verifying an appropriate error is still returned).