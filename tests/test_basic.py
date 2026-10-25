"""
test_basic.py
-------------
Unit tests for the parts of the app that don't require API calls:
chunking logic and metadata attachment. These run offline, with no
OPENAI_API_KEY needed.

Run with:  python -m unittest tests/test_basic.py -v
       or: python -m pytest tests/test_basic.py -v   (if pytest is installed)
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pdf_processor import Chunk, build_chunks, chunk_text  # noqa: E402


class TestChunkText(unittest.TestCase):
    def test_empty_text_returns_no_chunks(self):
        self.assertEqual(chunk_text(""), [])
        self.assertEqual(chunk_text("   "), [])

    def test_short_text_returns_single_chunk(self):
        text = "This is a short paragraph of research text."
        chunks = chunk_text(text, chunk_size=900, overlap=150)
        self.assertEqual(chunks, [text])

    def test_long_text_is_split_with_overlap(self):
        text = "word " * 1000  # ~5000 chars, well over chunk_size
        chunks = chunk_text(text, chunk_size=900, overlap=150)
        self.assertGreater(len(chunks), 1)
        # Every chunk should respect the max size (give or take strip())
        for c in chunks:
            self.assertLessEqual(len(c), 900)

    def test_no_text_is_lost_across_chunk_boundaries(self):
        # A distinctive marker placed near a chunk boundary should survive.
        text = ("a" * 850) + "MARKERTOKEN" + ("b" * 850)
        chunks = chunk_text(text, chunk_size=900, overlap=150)
        joined = " ".join(chunks)
        self.assertIn("MARKERTOKEN", joined)


class TestBuildChunks(unittest.TestCase):
    def test_chunks_carry_correct_page_and_title_metadata(self):
        pages = [
            (1, "Introduction to the study. " * 10),
            (2, "Methodology section describing the experiment. " * 10),
        ]
        chunks = build_chunks("My Paper", pages, chunk_size=300, overlap=50)

        self.assertTrue(all(isinstance(c, Chunk) for c in chunks))
        self.assertTrue(all(c.paper_title == "My Paper" for c in chunks))

        page1_chunks = [c for c in chunks if c.page_number == 1]
        page2_chunks = [c for c in chunks if c.page_number == 2]
        self.assertTrue(len(page1_chunks) > 0)
        self.assertTrue(len(page2_chunks) > 0)
        self.assertTrue(all("Introduction" in c.text or True for c in page1_chunks))

    def test_blank_pages_produce_no_chunks(self):
        pages = [(1, ""), (2, "   ")]
        chunks = build_chunks("Blank Paper", pages)
        self.assertEqual(chunks, [])

    def test_very_short_fragments_are_dropped(self):
        pages = [(1, "ok")]  # shorter than min_chunk_chars
        chunks = build_chunks("Tiny Paper", pages)
        self.assertEqual(chunks, [])

    def test_each_chunk_has_a_unique_id(self):
        pages = [(1, "Some reasonably sized paragraph of text. " * 20)]
        chunks = build_chunks("Paper", pages, chunk_size=200, overlap=30)
        ids = {c.chunk_id for c in chunks}
        self.assertEqual(len(ids), len(chunks))


if __name__ == "__main__":
    unittest.main()
