from retinal_ai.config import DRIVE_DIR
from retinal_ai.data import find_drive_pairs, split_items


def test_supplied_drive_archive_has_20_labelled_training_pairs():
    pairs = find_drive_pairs(DRIVE_DIR)
    assert len(pairs) == 20
    assert all(pair.image.exists() and pair.mask.exists() and pair.field_mask.exists() for pair in pairs)


def test_drive_dataset_returns_aligned_field_of_view_mask():
    from retinal_ai.data import DriveDataset

    image, truth, field_mask, identifier = DriveDataset(find_drive_pairs(DRIVE_DIR)[:1], 64)[0]
    assert image.shape == (3, 64, 64)
    assert truth.shape == field_mask.shape == (1, 64, 64)
    assert set(field_mask.unique().tolist()) <= {0.0, 1.0}
    assert identifier == "21"


def test_split_is_non_overlapping_and_deterministic():
    pairs = find_drive_pairs(DRIVE_DIR)
    one = split_items(pairs, 42)
    two = split_items(pairs, 42)
    assert [[item.identifier for item in split] for split in one] == [[item.identifier for item in split] for split in two]
    assert len({item.identifier for split in one for item in split}) == len(pairs)

