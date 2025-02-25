import pytest

from utils import format_float, ParabolicTrap, Trap, compute_parallel, param_phase_validator


def _fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

@pytest.fixture
def fibonacci_results():
    return ((10, 55), (20, 6765), (30, 832040), (40, 102334155), (50, 12586269025), (60, 1548008755920),
            (70, 190392490709135), (80, 23416728348467685), (90, 2880067194370816120), (100, 354224848179261915075))

def test_compute_parallel(fibonacci_results):
    results = compute_parallel(_fibonacci, [x for x, _ in fibonacci_results])
    assert results == [y for _, y in fibonacci_results]


@pytest.mark.parametrize("input_value, expected_output",
                         [
                             (1.23456789, "1.23"),
                             (4, "4.00"),
                             (1234.56789, "1.23 \\cdot 10^{3}"),
                         ]
                         )
def test_format_float(input_value, expected_output):
    assert format_float(input_value) == expected_output


@pytest.mark.parametrize("phi, expected", [
    (None, None), (5, -3), (1.0010000000001, 1.001)
])
def test_param_phase_validator(phi, expected):
    assert param_phase_validator(phi) == expected

def test_param_phase_validator_error():
    with pytest.raises(AssertionError):
        param_phase_validator(1.001000000001)


def test_parabolic_trap_equal():
    parabolic_trap = ParabolicTrap(depth=1, offset_relative=0)
    empty_parabolic_trap = ParabolicTrap(depth=0, offset_relative=.5)
    class EmptyTrap(Trap):
        def __bool__(self):
            return True
    assert parabolic_trap == ParabolicTrap(depth=1, offset_relative=0)
    assert parabolic_trap != ParabolicTrap(depth=.5, offset_relative=0)
    assert parabolic_trap != ParabolicTrap(depth=1, offset_relative=.5)
    assert parabolic_trap != empty_parabolic_trap
    assert parabolic_trap != EmptyTrap()
    assert parabolic_trap != None
    assert empty_parabolic_trap == None
