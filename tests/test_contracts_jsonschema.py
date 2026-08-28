"""GBNF built from a JSON Schema, checked as a grammar rather than as a string.

Every assertion about the text goes through ``rules_of``, which reads the grammar the way
llama.cpp does: rules on the left of ``::=``, and identifiers on the right that are not
inside a literal or a character class. A grammar naming a rule it never defines loads and
then rejects every token, so "the expected substring is present" is not the property worth
holding.
"""

from __future__ import annotations

import pytest
from ml_stack.contracts import ContractError, grammar_for

EXTRACTION = {
    "type": "object",
    "properties": {
        "people": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "maxLength": 80},
                    "role": {"type": "string"},
                    "location": {"type": "string"},
                },
                "required": ["name", "role", "location"],
            },
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "from": {"type": "string"},
                    "to": {"type": "string"},
                    "rel": {"type": "string",
                            "enum": ["reports-to", "colleague", "founder"]},
                },
            },
        },
    },
}


def identifiers(body: str) -> set[str]:
    """Rule names a body refers to, ignoring literals and character classes."""
    found: set[str] = set()
    i = 0
    while i < len(body):
        char = body[i]
        if char == '"':
            i += 1
            while i < len(body) and body[i] != '"':
                i += 2 if body[i] == "\\" else 1
            i += 1
        elif char == "[":
            i += 1
            while i < len(body) and body[i] != "]":
                i += 2 if body[i] == "\\" else 1
            i += 1
        elif char.isalpha():
            end = i
            while end < len(body) and (body[end].isalnum() or body[end] in "-_"):
                end += 1
            found.add(body[i:end])
            i = end
        else:
            i += 1
    return found


def rules_of(text: str) -> dict[str, str]:
    """Every ``name ::= body`` in a grammar. Raises on a line that is not one."""
    rules: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        name, sep, body = line.partition("::=")
        assert sep, f"not a rule: {line!r}"
        name = name.strip()
        assert name and identifiers(name) == {name}, f"bad rule name: {name!r}"
        assert name not in rules, f"{name} is defined twice"
        assert body.strip(), f"{name} has an empty body"
        rules[name] = body.strip()
    return rules


def referenced(rules: dict[str, str]) -> set[str]:
    return set().union(*(identifiers(body) for body in rules.values()))


class TestShape:
    def test_it_defines_every_rule_it_refers_to(self):
        rules = rules_of(grammar_for(EXTRACTION))
        assert referenced(rules) <= set(rules)

    def test_it_defines_nothing_it_never_refers_to(self):
        """An unused rule is a schema branch that silently did not make it in."""
        rules = rules_of(grammar_for(EXTRACTION))
        assert set(rules) - referenced(rules) == {"root"}

    def test_the_root_is_the_object(self):
        rules = rules_of(grammar_for(EXTRACTION))
        assert rules["root"] in rules
        assert rules[rules["root"]].startswith('"{"')

    def test_whitespace_is_the_json_set(self):
        assert rules_of(grammar_for(EXTRACTION))["ws"] == r"[ \t\n]*"

    def test_a_string_carries_the_escape_rule(self):
        rules = rules_of(grammar_for({"type": "object",
                                      "properties": {"a": {"type": "string"}}}))
        assert "char" in rules and "hex" in rules
        assert "char" in identifiers(rules["string"])


class TestDeterminism:
    def test_the_same_schema_gives_the_same_text(self):
        """A grammar that differs between calls invalidates the server's prompt cache
        and makes a failing extraction unreproducible."""
        assert grammar_for(EXTRACTION) == grammar_for(EXTRACTION)

    def test_two_equal_schemas_give_the_same_text(self):
        import copy

        assert grammar_for(copy.deepcopy(EXTRACTION)) == grammar_for(EXTRACTION)


class TestProperties:
    def test_every_property_is_emitted_in_order(self):
        rules = rules_of(grammar_for(EXTRACTION))
        person = next(body for body in rules.values() if r'"\"name\""' in body)
        assert person.index(r'"\"name\""') < person.index(r'"\"role\""')
        assert person.index(r'"\"role\""') < person.index(r'"\"location\""')

    def test_keys_are_quoted(self):
        rules = rules_of(grammar_for({"type": "object",
                                      "properties": {"a": {"type": "integer"}}}))
        assert r'"\"a\""' in rules["object-1"]

    def test_an_object_with_no_properties_is_the_empty_object(self):
        rules = rules_of(grammar_for({"type": "object", "properties": {}}))
        assert rules["object-1"] == '"{" ws "}"'

    def test_max_length_is_ignored(self):
        capped = {"type": "object", "properties": {"a": {"type": "string",
                                                         "maxLength": 3}}}
        plain = {"type": "object", "properties": {"a": {"type": "string"}}}
        assert grammar_for(capped) == grammar_for(plain)


class TestEnums:
    def test_the_values_appear_as_quoted_literals(self):
        text = grammar_for(EXTRACTION)
        for value in ("reports-to", "colleague", "founder"):
            assert f'"\\"{value}\\""' in text

    def test_it_is_an_alternation_of_only_those_values(self):
        rules = rules_of(grammar_for({"type": "string", "enum": ["a", "b"]}))
        assert rules["enum-1"] == r'"\"a\"" | "\"b\""'

    def test_an_empty_enum_is_refused(self):
        with pytest.raises(ContractError, match="non-empty"):
            grammar_for({"type": "string", "enum": []})


class TestRefs:
    def test_a_local_ref_resolves(self):
        schema = {
            "type": "object",
            "properties": {"here": {"$ref": "#/$defs/place"}},
            "$defs": {"place": {"type": "object",
                                "properties": {"city": {"type": "string"}}}},
        }
        rules = rules_of(grammar_for(schema))
        assert referenced(rules) <= set(rules)
        alias = next(name for name in rules if name.startswith("def-"))
        assert alias in identifiers(rules["object-1"])
        assert r'"\"city\""' in rules[rules[alias]]

    def test_the_same_ref_twice_is_one_rule(self):
        schema = {
            "type": "object",
            "properties": {"a": {"$ref": "#/$defs/x"}, "b": {"$ref": "#/$defs/x"}},
            "$defs": {"x": {"type": "string"}},
        }
        rules = rules_of(grammar_for(schema))
        assert len([name for name in rules if name.startswith("def-")]) == 1

    def test_definitions_is_read_as_well_as_defs(self):
        schema = {"type": "object", "properties": {"a": {"$ref": "#/definitions/x"}},
                  "definitions": {"x": {"type": "integer"}}}
        assert "integer" in rules_of(grammar_for(schema))

    def test_a_ref_to_nothing_says_what_it_knows(self):
        schema = {"type": "object", "properties": {"a": {"$ref": "#/$defs/absent"}},
                  "$defs": {"present": {"type": "string"}}}
        with pytest.raises(ContractError, match="present"):
            grammar_for(schema)

    def test_a_remote_ref_is_refused(self):
        with pytest.raises(ContractError, match="local"):
            grammar_for({"type": "object",
                         "properties": {"a": {"$ref": "https://example.com/s.json"}}})


class TestRefusals:
    @pytest.mark.parametrize("kind", ["date", "any", None, ["string", "null"]])
    def test_a_type_it_cannot_constrain_is_refused(self, kind):
        with pytest.raises(ContractError, match="unsupported schema type"):
            grammar_for({"type": "object", "properties": {"a": {"type": kind}}})

    def test_an_array_without_items_is_refused(self):
        with pytest.raises(ContractError, match="items"):
            grammar_for({"type": "object", "properties": {"a": {"type": "array"}}})

    def test_a_non_object_schema_is_refused(self):
        with pytest.raises(ContractError, match="not str"):
            grammar_for("object")
