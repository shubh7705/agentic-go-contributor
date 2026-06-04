"""
tools/tree_sitter_parser.py — Go source code parser using Tree-Sitter.

Builds a structured repository map:

{
    "binding/json.go": {
        "package": "binding",
        "imports": ["encoding/json", "net/http"],
        "functions": ["decodeJSON", "validate"],
        "methods": [{"receiver": "jsonBinding", "name": "Bind"}],
        "interfaces": ["Binding"],
        "structs": ["jsonBinding"],
        "types": ["BindingError"]
    }
}

Also provides per-file extraction utilities for targeted analysis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Tree-sitter imports — gracefully degrade if not installed
try:
    import tree_sitter_go as tsg
    from tree_sitter import Language, Parser, Node

    _GO_LANGUAGE = Language(tsg.language())
    _PARSER_AVAILABLE = True
    logger.debug("Tree-Sitter Go language loaded")
except Exception as exc:  # noqa: BLE001
    _PARSER_AVAILABLE = False
    logger.warning("Tree-Sitter not available (%s) — falling back to regex", exc)
    _GO_LANGUAGE = None  # type: ignore[assignment]
    Node = Any  # type: ignore[misc,assignment]


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class MethodInfo:
    receiver: str
    name: str
    params: list[str] = field(default_factory=list)
    return_types: list[str] = field(default_factory=list)


@dataclass
class FileMap:
    """Parsed structure for a single Go source file."""

    path: str
    package: str = ""
    imports: list[str] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)
    methods: list[MethodInfo] = field(default_factory=list)
    interfaces: list[str] = field(default_factory=list)
    structs: list[str] = field(default_factory=list)
    types: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "package": self.package,
            "imports": self.imports,
            "functions": self.functions,
            "methods": [
                {"receiver": m.receiver, "name": m.name} for m in self.methods
            ],
            "interfaces": self.interfaces,
            "structs": self.structs,
            "types": self.types,
        }


# ---------------------------------------------------------------------------
# Tree-Sitter parser
# ---------------------------------------------------------------------------

class GoParser:
    """
    Parses Go source files using Tree-Sitter.

    Falls back to regex-based extraction if Tree-Sitter is unavailable.
    """

    def __init__(self) -> None:
        if _PARSER_AVAILABLE:
            self._parser = Parser(_GO_LANGUAGE)
        else:
            self._parser = None

    # ── Public interface ─────────────────────────────────────────────────────

    def parse_file(self, path: str, source: str) -> FileMap:
        """
        Parse a single Go file and return its structured FileMap.

        Args:
            path:   Relative file path (used as identifier).
            source: Raw Go source code.

        Returns:
            FileMap with extracted symbols.
        """
        if self._parser is not None:
            return self._parse_with_tree_sitter(path, source)
        return self._parse_with_regex(path, source)

    def build_repo_map(
        self,
        file_contents: dict[str, str],
    ) -> dict[str, dict[str, Any]]:
        """
        Parse all files and return the full repository map.

        Args:
            file_contents: Mapping of {relative_path: source_code}.

        Returns:
            Mapping of {relative_path: file_map_dict}.
        """
        repo_map: dict[str, dict[str, Any]] = {}
        for path, source in file_contents.items():
            try:
                file_map = self.parse_file(path, source)
                repo_map[path] = file_map.to_dict()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to parse %s: %s", path, exc)
                repo_map[path] = {"package": "", "error": str(exc)}
        logger.info("Repository map built for %d files", len(repo_map))
        return repo_map

    # ── Tree-Sitter implementation ────────────────────────────────────────────

    def _parse_with_tree_sitter(self, path: str, source: str) -> FileMap:
        fm = FileMap(path=path)
        source_bytes = source.encode("utf-8")
        tree = self._parser.parse(source_bytes)
        root = tree.root_node

        for child in root.children:
            node_type = child.type

            if node_type == "package_clause":
                fm.package = self._text(child, source_bytes, "package_identifier") or ""

            elif node_type == "import_declaration":
                fm.imports.extend(self._extract_imports(child, source_bytes))

            elif node_type == "function_declaration":
                name = self._text(child, source_bytes, "identifier")
                if name:
                    fm.functions.append(name)

            elif node_type == "method_declaration":
                method = self._extract_method(child, source_bytes)
                if method:
                    fm.methods.append(method)

            elif node_type == "type_declaration":
                self._extract_type(child, source_bytes, fm)

        return fm

    def _text(
        self,
        node: "Node",
        source: bytes,
        child_type: Optional[str] = None,
    ) -> Optional[str]:
        """Extract text from a node or its first child of a given type."""
        if child_type is None:
            return source[node.start_byte: node.end_byte].decode("utf-8", errors="replace")
        for child in node.children:
            if child.type == child_type:
                return source[child.start_byte: child.end_byte].decode("utf-8", errors="replace")
        return None

    def _extract_imports(self, node: "Node", source: bytes) -> list[str]:
        imports: list[str] = []
        for child in node.children:
            if child.type in ("import_spec_list", "import_spec"):
                for spec in (child.children if child.type == "import_spec_list" else [child]):
                    if spec.type == "import_spec":
                        path_node = next(
                            (c for c in spec.children if c.type == "interpreted_string_literal"),
                            None,
                        )
                        if path_node:
                            raw = source[path_node.start_byte: path_node.end_byte].decode("utf-8")
                            imports.append(raw.strip('"'))
        return imports

    def _extract_method(self, node: "Node", source: bytes) -> Optional[MethodInfo]:
        receiver = ""
        name = ""
        for child in node.children:
            if child.type == "parameter_list" and receiver == "":
                # First parameter_list is the receiver
                for param in child.children:
                    if param.type == "parameter_declaration":
                        type_node = next(
                            (c for c in param.children if c.type in ("type_identifier", "pointer_type")),
                            None,
                        )
                        if type_node:
                            receiver = source[type_node.start_byte: type_node.end_byte].decode("utf-8")
            elif child.type == "field_identifier":
                name = source[child.start_byte: child.end_byte].decode("utf-8")
            elif child.type == "identifier" and name == "":
                name = source[child.start_byte: child.end_byte].decode("utf-8")
        if name:
            return MethodInfo(receiver=receiver, name=name)
        return None

    def _extract_type(self, node: "Node", source: bytes, fm: FileMap) -> None:
        for spec in node.children:
            if spec.type == "type_spec":
                name_node = next((c for c in spec.children if c.type == "type_identifier"), None)
                type_node = next(
                    (c for c in spec.children if c.type in ("struct_type", "interface_type")),
                    None,
                )
                if name_node:
                    name = source[name_node.start_byte: name_node.end_byte].decode("utf-8")
                    if type_node:
                        if type_node.type == "struct_type":
                            fm.structs.append(name)
                        elif type_node.type == "interface_type":
                            fm.interfaces.append(name)
                    else:
                        fm.types.append(name)

    # ── Regex fallback ────────────────────────────────────────────────────────

    def _parse_with_regex(self, path: str, source: str) -> FileMap:
        import re
        fm = FileMap(path=path)

        # Package
        m = re.search(r"^package\s+(\w+)", source, re.MULTILINE)
        if m:
            fm.package = m.group(1)

        # Imports
        for imp in re.findall(r'"([^"]+)"', source):
            if "/" in imp or imp.isidentifier():
                fm.imports.append(imp)

        # Functions (top-level)
        for m in re.finditer(r"^func\s+(\w+)\s*\(", source, re.MULTILINE):
            fm.functions.append(m.group(1))

        # Methods
        for m in re.finditer(
            r"^func\s+\(\s*\w+\s+\*?(\w+)\s*\)\s+(\w+)\s*\(", source, re.MULTILINE
        ):
            fm.methods.append(MethodInfo(receiver=m.group(1), name=m.group(2)))

        # Interfaces
        for m in re.finditer(r"^type\s+(\w+)\s+interface\s*\{", source, re.MULTILINE):
            fm.interfaces.append(m.group(1))

        # Structs
        for m in re.finditer(r"^type\s+(\w+)\s+struct\s*\{", source, re.MULTILINE):
            fm.structs.append(m.group(1))

        # Type aliases / other type decls
        for m in re.finditer(r"^type\s+(\w+)\s+(?!struct|interface)(\w+)", source, re.MULTILINE):
            fm.types.append(m.group(1))

        return fm


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_default_parser: Optional[GoParser] = None


def get_parser() -> GoParser:
    """Return the module-level GoParser singleton."""
    global _default_parser
    if _default_parser is None:
        _default_parser = GoParser()
    return _default_parser


def build_repo_map(file_contents: dict[str, str]) -> dict[str, dict[str, Any]]:
    """
    Build a repository map from a dict of {path: source} pairs.

    This is the primary public API for the repository agent.

    Args:
        file_contents: All Go source files to analyse.

    Returns:
        Repository map compatible with the AgentState.repository_map field.
    """
    return get_parser().build_repo_map(file_contents)
