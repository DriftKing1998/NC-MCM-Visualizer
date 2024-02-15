import unittest
from unittest.mock import MagicMock
from ncmcm.classes import *


class TestVisualizerMethods(unittest.TestCase):

    def setUp(self):
        # You can create a mock Database object for testing
        self.mock_database = Database(neuron_traces=[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                                     [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
                                                     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
                                      behavior=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                      neuron_names=['neuron1', 'neuron2', 'neuron3', 'neuron4'],
                                      fps=1)

        # Create a mock mapping for testing
        self.mock_mapping = PCA(n_components=3)

        # Initialize Visualizer with the mock Database and mapping
        self.visualizer = Visualizer(Data=self.mock_database, mapping=self.mock_mapping)

    def test_change_mapping_NMF(self):
        new_mapping = NMF(n_components=3)
        result = self.visualizer.change_mapping(new_mapping)
        self.assertTrue(result)
        self.assertEqual(self.visualizer.mapping, new_mapping)

    def test_change_mapping_TSNE(self):
        new_mapping = TSNE(n_components=3, perplexity=3)
        result = self.visualizer.change_mapping(new_mapping)
        self.assertTrue(result)
        self.assertEqual(self.visualizer.mapping, new_mapping)

    def test_change_mapping_failure(self):
        # If _transform_points fails, change_mapping should return False
        self.visualizer._transform_points = MagicMock(return_value=False)
        new_mapping = MagicMock()
        result = self.visualizer.change_mapping(new_mapping)
        self.assertFalse(result)
        self.assertEqual(self.visualizer.mapping, self.mock_mapping)

    def test_plot_mapping_success(self):
        # Assume transformed_points shape is (3, N)
        self.visualizer.transformed_points = np.random.rand(3, 10)
        result = self.visualizer.plot_mapping()
        self.assertTrue(result)

    def test_plot_mapping_failure(self):
        # If transformed_points shape is not (3, N), it should print an error message and return False
        self.visualizer.transformed_points = np.random.rand(2, 10)
        with unittest.mock.patch('builtins.print') as mock_print:
            result = self.visualizer.plot_mapping()
            mock_print.assert_called_once_with('The mapping does not map into a 3D space.')
        self.assertFalse(result)

    def test_make_comparison_more_transformed_points(self):
        self.visualizer.data.fit_model(LogisticRegression(max_iter=1000), binary=False)
        self.visualizer.transformed_points = np.random.rand(3, 20)
        with unittest.mock.patch('builtins.print') as mock_print:
            result = self.visualizer.make_comparison()
            mock_print.assert_any_call(
                'The prediction has fewer points than the true labels. Therefore 10 points are not plotted and also not used for accuracy calculation of the model')
        self.assertTrue(result)

        self.visualizer.data.fit_model(LogisticRegression(max_iter=1000), binary=True)
        self.visualizer.transformed_points = np.random.rand(3, 40)
        with unittest.mock.patch('builtins.print') as mock_print:
            result = self.visualizer.make_comparison()
            mock_print.assert_any_call(
                'The prediction has fewer points than the true labels. Therefore 30 points are not plotted and also not used for accuracy calculation of the model')
        self.assertTrue(result)

    def test_make_comparison_success(self):
        self.visualizer.data.fit_model(LogisticRegression(max_iter=1000), binary=False)
        self.visualizer.transformed_points = np.random.rand(3, 10)
        result = self.visualizer.make_comparison()
        self.assertTrue(result)

        self.visualizer.data.fit_model(LogisticRegression(max_iter=1000), binary=True)
        self.visualizer.transformed_points = np.random.rand(3, 10)
        result = self.visualizer.make_comparison()
        self.assertTrue(result)

    def test_make_comparison_failure(self):
        # If transformed_points shape is not (3, N), it should print an error message and return False
        self.visualizer.data.fit_model(LogisticRegression(max_iter=1000), binary=False)
        self.visualizer.transformed_points = np.random.rand(2, 10)
        with unittest.mock.patch('builtins.print') as mock_print:
            result = self.visualizer.make_comparison()
            mock_print.assert_called_once_with('The mapping does not map into a 3D space.')
        self.assertFalse(result)

        # If transformed_points shape is not (3, N), it should print an error message and return False
        self.visualizer.data.fit_model(LogisticRegression(max_iter=1000), binary=True)
        self.visualizer.transformed_points = np.random.rand(2, 10)
        with unittest.mock.patch('builtins.print') as mock_print:
            result = self.visualizer.make_comparison()
            mock_print.assert_called_once_with('The mapping does not map into a 3D space.')
        self.assertFalse(result)

    def test_make_movie_success(self):
        result = self.visualizer.make_movie()
        self.assertTrue(result)

        result = self.visualizer.make_movie(quivers=True, show_legend=True, grid_off=False)
        self.assertTrue(result)

    def test_make_movie_failure(self):
        self.visualizer.transformed_points = np.random.rand(2, 10)
        with unittest.mock.patch('builtins.print') as mock_print:
            result = self.visualizer.make_movie()
            mock_print.assert_called_once_with('The mapping does not map into a 3D space.')
        self.assertFalse(result)

        with unittest.mock.patch('builtins.print') as mock_print:
            result = self.visualizer.make_movie(quivers=True, show_legend=True, grid_off=False)
            mock_print.assert_called_once_with('The mapping does not map into a 3D space.')
        self.assertFalse(result)

    # You can write similar tests for other methods

    def tearDown(self):
        # Clean up any resources or configurations used in the tests
        pass


if __name__ == '__main__':
    unittest.main()
