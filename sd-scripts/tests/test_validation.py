from library.train_util import split_train_val


def test_split_train_val():
    paths = ["path1", "path2", "path3", "path4", "path5", "path6", "path7"]
    sizes = [(1, 1), (2, 2), None, (4, 4), (5, 5), (6, 6), None]
    result_paths, result_sizes = split_train_val(paths, sizes, True, 0.2, 1234)
    assert result_paths == ["path2", "path3", "path6", "path5", "path1", "path4"], result_paths
    assert result_sizes == [(2, 2), None, (6, 6), (5, 5), (1, 1), (4, 4)], result_sizes

    result_paths, result_sizes = split_train_val(paths, sizes, False, 0.2, 1234)
    assert result_paths == ["path7"], result_paths
    assert result_sizes == [None], result_sizes


if __name__ == "__main__":
    test_split_train_val()
