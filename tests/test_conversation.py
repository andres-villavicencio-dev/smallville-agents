import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
from datetime import datetime

from conversation import Conversation, ConversationManager, ConversationTurn


class TestConversation(unittest.TestCase):
    """Test Conversation class pure logic (synchronous)."""

    def test_init_defaults(self):
        """Conversation initializes with correct defaults."""
        conv = Conversation("Alice", "Bob", "Park")
        self.assertTrue(conv.active)
        self.assertEqual(conv.max_turns, 8)
        self.assertEqual(len(conv.turns), 0)
        self.assertEqual(conv.agent1, "Alice")
        self.assertEqual(conv.agent2, "Bob")
        self.assertEqual(conv.location, "Park")
        self.assertIsInstance(conv.start_time, datetime)

    def test_add_turn(self):
        """add_turn adds ConversationTurn to turns list."""
        conv = Conversation("Alice", "Bob", "Park")
        conv.add_turn("Alice", "Hello Bob!")
        conv.add_turn("Bob", "Hi Alice!")

        self.assertEqual(len(conv.turns), 2)
        self.assertEqual(conv.turns[0].speaker, "Alice")
        self.assertEqual(conv.turns[0].message, "Hello Bob!")
        self.assertEqual(conv.turns[1].speaker, "Bob")
        self.assertEqual(conv.turns[1].message, "Hi Alice!")

    def test_get_history_text(self):
        """get_history_text formats turns as 'Speaker: message' lines."""
        conv = Conversation("Alice", "Bob", "Park")
        conv.add_turn("Alice", "How are you?")
        conv.add_turn("Bob", "I'm good, thanks!")

        history = conv.get_history_text()
        expected = "Alice: How are you?\nBob: I'm good, thanks!"
        self.assertEqual(history, expected)

    def test_should_end_under_max(self):
        """should_end returns False when turns < max_turns."""
        conv = Conversation("Alice", "Bob", "Park")
        conv.add_turn("Alice", "Hello")
        conv.add_turn("Bob", "Hi")

        self.assertFalse(conv.should_end())

    def test_should_end_at_max(self):
        """should_end returns True when turns >= max_turns."""
        conv = Conversation("Alice", "Bob", "Park")
        for i in range(8):
            speaker = "Alice" if i % 2 == 0 else "Bob"
            conv.add_turn(speaker, f"Message {i}")

        self.assertTrue(conv.should_end())

    def test_should_end_inactive(self):
        """should_end returns True when active=False regardless of turn count."""
        conv = Conversation("Alice", "Bob", "Park")
        conv.add_turn("Alice", "Hello")
        conv.active = False

        self.assertTrue(conv.should_end())

    def test_get_participants(self):
        """get_participants returns (agent1, agent2) tuple."""
        conv = Conversation("Alice", "Bob", "Park")
        participants = conv.get_participants()

        self.assertIsInstance(participants, tuple)
        self.assertEqual(participants, ("Alice", "Bob"))


class TestConversationManager(unittest.TestCase):
    """Test ConversationManager pure logic (synchronous)."""

    def test_key_sorted(self):
        """get_conversation_key returns sorted tuple."""
        manager = ConversationManager()
        key = manager.get_conversation_key("Bob", "Alice")

        self.assertEqual(key, ("Alice", "Bob"))
        self.assertIsInstance(key, tuple)

    def test_key_already_sorted(self):
        """get_conversation_key handles already sorted names."""
        manager = ConversationManager()
        key = manager.get_conversation_key("Alice", "Bob")

        self.assertEqual(key, ("Alice", "Bob"))

    def test_has_active_false(self):
        """has_active_conversation returns False for empty manager."""
        manager = ConversationManager()

        self.assertFalse(manager.has_active_conversation("Alice", "Bob"))

    def test_has_active_true(self):
        """has_active_conversation returns True when conversation exists."""
        manager = ConversationManager()
        conv = Conversation("Alice", "Bob", "Park")
        key = manager.get_conversation_key("Alice", "Bob")
        manager.active_conversations[key] = conv

        self.assertTrue(manager.has_active_conversation("Alice", "Bob"))
        # Test reverse order
        self.assertTrue(manager.has_active_conversation("Bob", "Alice"))

    def test_get_active_none(self):
        """get_active_conversation returns None for unknown pair."""
        manager = ConversationManager()

        result = manager.get_active_conversation("Alice", "Bob")
        self.assertIsNone(result)

    def test_get_active_returns_conversation(self):
        """get_active_conversation retrieves stored conversation."""
        manager = ConversationManager()
        conv = Conversation("Alice", "Bob", "Park")
        key = manager.get_conversation_key("Alice", "Bob")
        manager.active_conversations[key] = conv

        retrieved = manager.get_active_conversation("Alice", "Bob")
        self.assertIs(retrieved, conv)
        self.assertEqual(retrieved.location, "Park")

    def test_summary_empty(self):
        """get_active_conversations_summary returns empty list when no conversations."""
        manager = ConversationManager()

        summary = manager.get_active_conversations_summary()
        self.assertEqual(summary, [])
        self.assertIsInstance(summary, list)

    def test_summary_format(self):
        """get_active_conversations_summary formats conversations correctly."""
        manager = ConversationManager()
        conv = Conversation("Alice", "Bob", "Park")
        conv.add_turn("Alice", "Hello")
        conv.add_turn("Bob", "Hi")

        key = manager.get_conversation_key("Alice", "Bob")
        manager.active_conversations[key] = conv

        summary = manager.get_active_conversations_summary()
        self.assertEqual(len(summary), 1)

        # Verify format contains key information
        summary_str = summary[0]
        self.assertIn("Alice", summary_str)
        self.assertIn("Bob", summary_str)
        self.assertIn("Park", summary_str)
        self.assertIn("2", summary_str)  # 2 turns


if __name__ == '__main__':
    unittest.main()
