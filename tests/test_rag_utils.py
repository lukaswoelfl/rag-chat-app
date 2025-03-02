import os

import pytest

from rag_utils import init_rag_system, load_all_pdfs_from_directory, load_and_split_pdf


def test_load_and_split_pdf_file_not_found():
    with pytest.raises(FileNotFoundError):
        # Since init_rag_system handles non-existent paths
        init_rag_system("non_existent.pdf")


def test_load_all_pdfs_from_directory_invalid():
    with pytest.raises(NotADirectoryError):
        load_all_pdfs_from_directory("non_existent_folder")
