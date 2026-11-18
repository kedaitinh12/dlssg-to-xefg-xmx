"""Regression for the +8 ProgramAsset pointer bug; no GPU/Windows required."""
import importlib.util
from pathlib import Path
import struct
import unittest

spec = importlib.util.spec_from_file_location('extractor', Path(__file__).resolve().parents[2] / 'tools/ExtractXeFgVulkanPrograms.py')
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)

class Image:
    def __init__(self):
        self.data = bytearray(512)
    def va_to_offset(self, va):
        return va

class ProgramRecords(unittest.TestCase):
    def image(self):
        img = Image()
        for at, char in [(160, 'a'), (225, 'b'), (290, 'c'), (355, 'd')]:
            img.data[at:at+65] = (char*64).encode() + b'\0'
        # source, resource, pad, SPIR-V hash, bytes, push, binding start/count
        struct.pack_into('<QIIQIIII', img.data, 32, 160, 7400, 0, 290, 4088, 16, 0, 3)
        struct.pack_into('<QIIQIIII', img.data, 72, 225, 7401, 0, 355, 8192, 48, 3, 6)
        return img
    def test_first_and_last_source_belong_to_same_record(self):
        rows = e.find_program_table(self.image(), [7400, 7401])
        self.assertEqual([r['source'] for r in rows], ['a'*64, 'b'*64])
        self.assertEqual([r['sha256'] for r in rows], ['c'*64, 'd'*64])
        self.assertEqual([r['bind_start'] for r in rows], [0, 3])
    def test_null_source_does_not_borrow_next_row(self):
        img = self.image()
        struct.pack_into('<Q', img.data, 32, 0)
        rows = e.find_program_table(img, [7400, 7401])
        self.assertIsNone(rows[0]['source'])
        self.assertEqual(rows[1]['source'], 'b'*64)
    def test_bad_source_is_rejected(self):
        img = self.image()
        img.data[160] = ord('z')
        with self.assertRaises(e.ExtractError):
            e.find_program_table(img, [7400, 7401])

if __name__ == '__main__':
    unittest.main()
