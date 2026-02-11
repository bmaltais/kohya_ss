# Tests

## Install

```
pip install pytest
```

## Usage

```
pytest
```

## Contribution

Pytest is configured to run tests in this directory. It might be a good idea to add tests closer in the code, as well as doctests.

Tests are functions starting with `test_` and files with the pattern `test_*.py`.

```
def test_x():
    assert 1 == 2, "Invalid test response"
```

## Resources

### pytest 

- https://docs.pytest.org/en/stable/index.html
- https://docs.pytest.org/en/stable/how-to/assert.html
- https://docs.pytest.org/en/stable/how-to/doctest.html

### PyTorch testing

- https://circleci.com/blog/testing-pytorch-model-with-pytest/
- https://pytorch.org/docs/stable/testing.html
- https://github.com/pytorch/pytorch/wiki/Running-and-writing-tests
- https://github.com/huggingface/pytorch-image-models/tree/main/tests
- https://github.com/pytorch/pytorch/tree/main/test

