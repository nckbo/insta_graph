import unittest

class TestMutuallyExclusivePairs(unittest.TestCase):

    def test_basic(self):
        self.assertEqual(
            mutually_exclusive_pairs([1, 2, 3]),
            [([1], [2, 3]), ([2], [1, 3]), ([3], [1, 2])]
        )

    def test_empty_list(self):
        self.assertEqual(mutually_exclusive_pairs([]), [])

    def test_single_element(self):
        self.assertEqual(mutually_exclusive_pairs([5]), [])

    def test_two_elements(self):
        self.assertEqual(mutually_exclusive_pairs([5, 10]), [([5], [10])])

    def test_four_elements(self):
        result = mutually_exclusive_pairs([1, 2, 3, 4])
        expected = [
            ([1], [2, 3, 4]),
            ([2], [1, 3, 4]),
            ([3], [1, 2, 4]),
            ([4], [1, 2, 3]),
            ([1, 2], [3, 4]),
            ([1, 3], [2, 4]),
            ([1, 4], [2, 3])
        ]

        # Convert lists to sets for unordered comparison
        result_sets = {frozenset((tuple(x[0]), tuple(x[1]))) for x in result}
        expected_sets = {frozenset((tuple(x[0]), tuple(x[1]))) for x in expected}

        self.assertEqual(result_sets, expected_sets)


class TestFlattenListOfListOfFsets(unittest.TestCase):

    def test_valid_input(self):
        input_data = [[frozenset([1, 2]), frozenset([3, 4])], [frozenset([5])]]
        expected_output = [{1, 2, 3, 4}, {5}]
        self.assertEqual(flatten_list_of_list_of_fsets(input_data), expected_output)

    def test_input_not_list_or_tuple(self):
        with self.assertRaises(TypeError):
            flatten_list_of_list_of_fsets(set([frozenset([1, 2]), frozenset([3, 4])]))

    def test_input_inner_not_list(self):
        with self.assertRaises(TypeError):
            flatten_list_of_list_of_fsets([(frozenset([1, 2]), frozenset([3, 4]))])

    def test_input_inner_list_not_frozenset(self):
        with self.assertRaises(TypeError):
            flatten_list_of_list_of_fsets([[{1, 2}, {3, 4}]])

    def test_empty_input(self):
        self.assertEqual(flatten_list_of_list_of_fsets([]), [])

    def test_nested_empty_list(self):
        self.assertEqual(flatten_list_of_list_of_fsets([[]]), [set()])

import pytest
import networkx as nx
from typing import FrozenSet


class TestGetSplit:

    @pytest.fixture
    def karate_graph(self):
        return nx.karate_club_graph()

    # AssertionError Tests
    def test_groups_to_consider_not_list(self, karate_graph):
        with pytest.raises(AssertionError, match="groups_to_consider must be a list"):
            get_split(karate_graph, "not a list")

    def test_group_in_groups_to_consider_not_frozenset(self, karate_graph):
        with pytest.raises(AssertionError, match="Each group in groups_to_consider must be a Frozenset"):
            get_split(karate_graph, [set([1, 2, 3])])

    def test_other_groups_not_list(self, karate_graph):
        with pytest.raises(AssertionError, match="other_groups must be a list"):
            get_split(karate_graph, [frozenset([1, 2, 3])], "not a list")

    def test_group_in_other_groups_not_frozenset(self, karate_graph):
        with pytest.raises(AssertionError, match="Each group in other_groups must be a Frozenset"):
            get_split(karate_graph, [frozenset([1, 2, 3])], [set([4, 5, 6])])

    # Testing Core Functionality
    @pytest.mark.parametrize(
        "groups_to_consider, other_groups, expected",
        [
            ( # Test Case 1
                [frozenset({0, 1, 2, 3, 7, 11, 12, 13, 17, 19, 21}), frozenset({4, 5, 6, 10, 16}), frozenset({8, 9, 14, 15, 18, 20, 22, 23, 26, 27, 29, 30, 32, 33}), frozenset({24, 25, 28, 31})],
                None,
                ([frozenset({0, 1, 2, 3, 7, 11, 12, 13, 17, 19, 21}), frozenset({4, 5, 6, 10, 16})], [frozenset({24, 25, 28, 31}), frozenset({8, 9, 14, 15, 18, 20, 22, 23, 26, 27, 29, 30, 32, 33})])
            ),
            (  # Test Case 2
                [frozenset({0, 1, 2, 3, 7, 11, 12, 13, 17, 19, 21}), frozenset({4, 5, 6, 10, 16}), frozenset({24, 25, 28, 31})],
                [frozenset({8, 9, 14, 15, 18, 20, 22, 23, 26, 27, 29, 30, 32, 33})],
                ([frozenset({0, 1, 2, 3, 7, 11, 12, 13, 17, 19, 21})], [frozenset({24, 25, 28, 31}), frozenset({4, 5, 6, 10, 16})])
            ),
            (  # Test Case 3
                [frozenset({0, 1, 2, 3, 7, 11, 12, 13, 17, 19, 21}), frozenset({4, 5, 6, 10, 16})],
                [frozenset({24, 25, 28, 31}), frozenset({8, 9, 14, 15, 18, 20, 22, 23, 26, 27, 29, 30, 32, 33})],
                ([frozenset({0, 1, 2, 3, 7, 11, 12, 13, 17, 19, 21})], [frozenset({4, 5, 6, 10, 16})])
            )
        ]
    )
    def test_get_split_functionality(self, karate_graph, groups_to_consider, other_groups, expected):
        result = get_split(karate_graph, groups_to_consider, other_groups)

        # Make a copy of the expected lists so we can remove items without affecting the original.
        expected_copy = list(expected)

        # Check that each list within the tuple has an equivalent list in the other tuple
        for r_list in result:
            matched = False
            for idx, e_list in enumerate(expected_copy):
                if set(r_list) == set(e_list):
                    matched = True
                    del expected_copy[idx]  # Remove the matched item
                    break

            assert matched, f"Expected: {expected}, but got: {result}"

        # Ensure that all expected lists have been matched
        assert not expected_copy, f"Some expected groups were not found in result: {expected_copy}"

    def test_no_duplicate_elements_in_output(self, karate_graph):
        groups_to_consider = [frozenset({0, 1, 2, 3, 7, 11, 12, 13, 17, 19, 21}), frozenset({4, 5, 6, 10, 16})]
        other_groups = [frozenset({24, 25, 28, 31}), frozenset({8, 9, 14, 15, 18, 20, 22, 23, 26, 27, 29, 30, 32, 33})]

        result = get_split(karate_graph, groups_to_consider, other_groups)

        # Flatten the output to check for duplicates
        all_elements = [item for fset in result for item in fset]
        unique_elements = set(all_elements)

        # Assert no duplicates
        assert len(all_elements) == len(unique_elements), f"Duplicates found in result: {result}"


class TestInitializeAndSplit:

    @pytest.fixture
    def karate_graph(self):
        return nx.karate_club_graph()

    @pytest.fixture
    def partitions(self):
        return [frozenset({0, 1, 2, 3, 7, 11, 12, 13, 17, 19, 21}),
                frozenset({4, 5, 6, 10, 16}),
                frozenset({8, 9, 14, 15, 18, 20, 22, 23, 26, 27, 29, 30, 32, 33}),
                frozenset({24, 25, 28, 31})]

    def test_elements_coverage(self, karate_graph, partitions):
        first_split, _ = initialize_and_split(karate_graph, partitions)

        all_elements = [item for sublist in first_split for fset in sublist for item in fset]
        assert set(all_elements) == set(karate_graph.nodes), f"Expected coverage of all nodes but got {set(all_elements)}"

    def test_first_split_type(self, karate_graph, partitions):
        first_split, _ = initialize_and_split(karate_graph, partitions)

        assert isinstance(first_split, tuple), f"Expected a tuple but got {type(first_split)}"
        assert len(first_split) == 2, f"Expected tuple of length 2 but got length {len(first_split)}"
        for lst in first_split:
            assert isinstance(lst, list), f"Expected a list within the tuple but got {type(lst)}"
            for fs in lst:
                assert isinstance(fs, FrozenSet), f"Expected a frozenset within the list but got {type(fs)}"

    def test_lists_to_split_type(self, karate_graph, partitions):
        _, lists_to_split = initialize_and_split(karate_graph, partitions)

        assert isinstance(lists_to_split, list), f"Expected a list but got {type(lists_to_split)}"
        for lst in lists_to_split:
            assert isinstance(lst, list), f"Expected a list within the main list but got {type(lst)}"
            for fs in lst:
                assert isinstance(fs, FrozenSet), f"Expected a frozenset within the list but got {type(fs)}"


    # def test_modularity_calculation(self, karate_graph):
    # # You'd typically need known splits and their modularities to test this.
    # # For demonstration purposes, let's use a hypothetical split:
    #     groups_to_consider = [frozenset(range(10)), frozenset(range(10, 20))]
    # # Now, you would compare the output of get_split to the expected best split.
    # # This requires domain knowledge or known good values.
    #
    # def test_integration_with_helpers(self, karate_graph):
    #     # Here, you'd mock the return values of `mutually_exclusive_pairs` and
    #     # `flatten_list_of_list_of_fsets` to specific values, then check that
    #     # `get_split` handles those mock values correctly.
    #     pass

if __name__ == '__main__':
    unittest.main()
