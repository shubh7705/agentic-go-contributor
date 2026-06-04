"""
tests/test_tools.py — Unit tests for the tools layer.

Tests are designed to run without network access or a real Go repository.
Ripgrep tests are skipped if `rg` is not installed.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# file_editor tests
# ---------------------------------------------------------------------------

class TestFileEditor:
    def test_generate_diff_detects_changes(self):
        from tools.file_editor import generate_diff

        original = "line1\nline2\nline3\n"
        modified = "line1\nline2 changed\nline3\n"
        diff = generate_diff(original, modified, "test.go")
        assert "-line2" in diff
        assert "+line2 changed" in diff

    def test_generate_diff_no_changes(self):
        from tools.file_editor import generate_diff

        content = "same content\n"
        diff = generate_diff(content, content, "test.go")
        assert diff == ""

    def test_replace_content_writes_file(self, tmp_path):
        from tools.file_editor import replace_content

        rel = "pkg/foo.go"
        content = "package foo\n\nfunc Foo() {}\n"
        replace_content(tmp_path, rel, content)

        assert (tmp_path / rel).exists()
        assert (tmp_path / rel).read_text() == content

    def test_replace_content_returns_diff(self, tmp_path):
        from tools.file_editor import replace_content

        rel = "pkg/bar.go"
        original = "package bar\n\nfunc Bar() {}\n"
        (tmp_path / "pkg").mkdir()
        (tmp_path / rel).write_text(original)

        modified = "package bar\n\nfunc Bar() { return }\n"
        diff = replace_content(tmp_path, rel, modified)
        assert "-func Bar() {}" in diff
        assert "+func Bar() { return }" in diff

    def test_extract_code_blocks(self):
        from tools.file_editor import extract_code_blocks

        response = textwrap.dedent("""\
            Here is the fix:

            // File: binding/json.go
            ```go
            package binding

            func Bind() {}
            ```

            That's it!
        """)
        blocks = extract_code_blocks(response)
        assert "binding/json.go" in blocks
        assert "func Bind()" in blocks["binding/json.go"]

    def test_extract_diff_blocks(self):
        from tools.file_editor import extract_diff_blocks

        response = textwrap.dedent("""\
            ```diff
            -old line
            +new line
            ```
        """)
        diff = extract_diff_blocks(response)
        assert "-old line" in diff
        assert "+new line" in diff


# ---------------------------------------------------------------------------
# git tools tests
# ---------------------------------------------------------------------------

class TestGitTools:
    def test_list_go_files_finds_go_files(self, tmp_path):
        from tools.git import list_go_files

        # Create some .go files
        (tmp_path / "main.go").write_text("package main\n")
        pkg_dir = tmp_path / "pkg"
        pkg_dir.mkdir()
        (pkg_dir / "foo.go").write_text("package pkg\n")
        # A non-go file that should be ignored
        (tmp_path / "readme.txt").write_text("readme")
        # vendor dir should be skipped
        vendor = tmp_path / "vendor" / "ext"
        vendor.mkdir(parents=True)
        (vendor / "ext.go").write_text("package ext\n")

        files = list_go_files(tmp_path)
        paths = [f.name for f in files]
        assert "main.go" in paths
        assert "foo.go" in paths
        assert "ext.go" not in paths  # vendor skipped

    def test_read_file(self, tmp_path):
        from tools.git import read_file

        (tmp_path / "hello.go").write_text("package main\n")
        content = read_file(tmp_path, "hello.go")
        assert content == "package main\n"

    def test_read_file_missing_raises(self, tmp_path):
        from tools.git import read_file

        with pytest.raises(FileNotFoundError):
            read_file(tmp_path, "nonexistent.go")

    def test_write_file(self, tmp_path):
        from tools.git import write_file

        write_file(tmp_path, "sub/new.go", "package sub\n")
        assert (tmp_path / "sub" / "new.go").read_text() == "package sub\n"


# ---------------------------------------------------------------------------
# tree_sitter_parser tests
# ---------------------------------------------------------------------------

class TestTreeSitterParser:
    _SAMPLE_GO = textwrap.dedent("""\
        package binding

        import (
            "encoding/json"
            "net/http"
        )

        type Binding interface {
            Bind(*http.Request, interface{}) error
        }

        type jsonBinding struct{}

        func (jsonBinding) Bind(req *http.Request, obj interface{}) error {
            return decodeJSON(req.Body, obj)
        }

        func decodeJSON(r interface{}, obj interface{}) error {
            return json.NewDecoder(r.(interface{ Read([]byte) (int, error) })).Decode(obj)
        }
    """)

    def test_parse_package(self):
        from tools.tree_sitter_parser import GoParser

        parser = GoParser()
        fm = parser.parse_file("binding/json.go", self._SAMPLE_GO)
        assert fm.package == "binding"

    def test_parse_interfaces(self):
        from tools.tree_sitter_parser import GoParser

        parser = GoParser()
        fm = parser.parse_file("binding/json.go", self._SAMPLE_GO)
        assert "Binding" in fm.interfaces

    def test_parse_structs(self):
        from tools.tree_sitter_parser import GoParser

        parser = GoParser()
        fm = parser.parse_file("binding/json.go", self._SAMPLE_GO)
        assert "jsonBinding" in fm.structs

    def test_parse_functions(self):
        from tools.tree_sitter_parser import GoParser

        parser = GoParser()
        fm = parser.parse_file("binding/json.go", self._SAMPLE_GO)
        assert "decodeJSON" in fm.functions

    def test_build_repo_map(self):
        from tools.tree_sitter_parser import build_repo_map

        repo_map = build_repo_map({"binding/json.go": self._SAMPLE_GO})
        assert "binding/json.go" in repo_map
        meta = repo_map["binding/json.go"]
        assert meta["package"] == "binding"
        assert isinstance(meta["functions"], list)


# ---------------------------------------------------------------------------
# GitHub URL parsing tests
# ---------------------------------------------------------------------------

class TestGitHubParsing:
    def test_parse_issue_url(self):
        from tools.github import parse_issue_url

        owner, repo, num = parse_issue_url(
            "https://github.com/gin-gonic/gin/issues/1234"
        )
        assert owner == "gin-gonic"
        assert repo == "gin"
        assert num == 1234

    def test_parse_issue_url_invalid(self):
        from tools.github import parse_issue_url

        with pytest.raises(ValueError):
            parse_issue_url("https://github.com/gin-gonic/gin")

    def test_parse_repo_url(self):
        from tools.github import parse_repo_url

        owner, repo = parse_repo_url("https://github.com/gin-gonic/gin")
        assert owner == "gin-gonic"
        assert repo == "gin"

    def test_parse_repo_url_with_git_suffix(self):
        from tools.github import parse_repo_url

        owner, repo = parse_repo_url("https://github.com/gin-gonic/gin.git")
        assert repo == "gin"


# ---------------------------------------------------------------------------
# Search tests (skip if rg not installed)
# ---------------------------------------------------------------------------

rg_available = shutil.which("rg") is not None

@pytest.mark.skipif(not rg_available, reason="ripgrep not installed")
class TestSearch:
    def test_search_text_finds_matches(self, tmp_path):
        from tools.search import search_text

        go_file = tmp_path / "main.go"
        go_file.write_text("package main\n\nfunc main() {\n\tprintln(\"hello\")\n}\n")

        results = search_text(tmp_path, "println")
        assert len(results.matches) > 0
        assert "main.go" in results.matched_files

    def test_search_text_no_matches(self, tmp_path):
        from tools.search import search_text

        go_file = tmp_path / "main.go"
        go_file.write_text("package main\n")

        results = search_text(tmp_path, "nonexistentXYZ")
        assert len(results.matches) == 0

    def test_search_symbol_func(self, tmp_path):
        from tools.search import search_symbol

        go_file = tmp_path / "util.go"
        go_file.write_text("package util\n\nfunc MyFunc() {}\n")

        results = search_symbol(tmp_path, "MyFunc", kinds=["func"])
        assert len(results.matches) > 0

    def test_search_package(self, tmp_path):
        from tools.search import search_package

        go_file = tmp_path / "foo.go"
        go_file.write_text("package mypackage\n")

        results = search_package(tmp_path, "mypackage")
        assert "foo.go" in results.matched_files
