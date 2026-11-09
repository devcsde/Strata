"""Local fix (Rakazo): with `use_tool` declared, a direct call of an undeclared tool goes out as use_tool."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from serve.frontend import OutputParser  # noqa: E402


def tool(name, props=None):
    return {"name": name, "description": "d", "parameters": {"type": "object", "properties": props or {}}}


RAKAZO = [tool("shell", {"command": {"type": "string"}}), tool("find_tool"),
          tool("use_tool", {"name": {"type": "string"}, "arguments": {"type": "object"}})]
OTHER = [tool("shell", {"command": {"type": "string"}})]

TEXT = ("</think>B und C sind bereits auf origin.\n\n"
        "<tool_call>\n<function=message_bot>\n<parameter=confirm_name>\nREVIEW\n</parameter>\n"
        "<parameter=message>\nB+C auf origin verifiziert.\n</parameter>\n</function>\n</tool_call>\n"
        "<tool_call>\n<function=shell>\n<parameter=command>\ngit status\n</parameter>\n</function>\n</tool_call>")


def run(tools, chunk):
    p = OutputParser(thinking=True, tools=tools, stream_tools=True, recover=True)
    ev = []
    for i in range(0, len(TEXT), chunk):
        ev += p.feed(TEXT[i:i + chunk])
    ev += p.finish("stop")
    content = "".join(e.text for e in ev if e.kind == "content")
    calls = [(e.call.name, e.call.arguments) for e in ev if e.kind == "tool_call"]
    return p, content, calls


class UseToolWrap(unittest.TestCase):
    def test_undeclared_call_goes_out_as_use_tool(self):
        for chunk in (1, 7, len(TEXT)):
            p, content, calls = run(RAKAZO, chunk)
            self.assertEqual(content, "B und C sind bereits auf origin.")
            self.assertEqual(calls, [
                ("use_tool", {"name": "message_bot",
                              "arguments": {"confirm_name": "REVIEW", "message": "B+C auf origin verifiziert."}}),
                ("shell", {"command": "git status"})])
            self.assertEqual((p.unread, p.wrapped), (0, ["message_bot"]))

    def test_without_use_tool_unchanged(self):
        for chunk in (1, len(TEXT)):
            p, content, calls = run(OTHER, chunk)
            self.assertIn("<function=message_bot>", content)
            self.assertEqual(calls, [("shell", {"command": "git status"})])
            self.assertEqual((p.unread, p.wrapped), (1, []))


if __name__ == "__main__":
    unittest.main()
