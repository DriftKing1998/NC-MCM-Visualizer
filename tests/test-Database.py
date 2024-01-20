import unittest
from unittest.mock import MagicMock

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
import networkx as nx
import matplotlib.pyplot as plt
from scripts.classes import *


class TestDatabase(unittest.TestCase):

    def setUp(self):
        # Set up any necessary data or configurations for the tests
        pass

    def test_init(self):
        # Test the initialization of the Database class
        # Include cases for both default and custom parameter values

        # Example with default parameters
        db_default = Database(neuron_traces=[[1, 2], [3, 4]], behavior=[0, 1])
        self.assertTrue(np.array_equal(db_default.neuron_traces, np.array([[1, 2], [3, 4]])))
        self.assertIsNone(db_default.fps)
        self.assertEqual(db_default.name, 'nc-mcm')
        self.assertTrue(np.array_equal(db_default.B, np.array([0, 1])))
        self.assertIsNotNone(db_default.states)
        self.assertTrue(np.array_equal(db_default.neuron_names, np.array(['0', '1'])))
        self.assertIsNone(db_default.pred_model)

        # Example with custom parameters
        neuron_traces_custom = np.array([[1, 2], [4, 5]])
        behavior_custom = np.array([1, 0])
        neuron_names_custom = np.array(['neuron1', 'neuron2'])
        states_custom = np.array(['state0', 'state1'])
        fps_custom = 30.0
        name_custom = 'custom-name'

        db_custom = Database(
            neuron_traces=neuron_traces_custom,
            behavior=behavior_custom,
            neuron_names=neuron_names_custom,
            states=states_custom,
            fps=fps_custom,
            name=name_custom
        )

        self.assertTrue(np.array_equal(db_custom.neuron_traces, neuron_traces_custom))
        self.assertEqual(db_custom.fps, fps_custom)
        self.assertEqual(db_custom.name, name_custom)
        self.assertTrue(np.array_equal(db_custom.B, behavior_custom))
        self.assertTrue(np.array_equal(db_custom.states, states_custom))
        self.assertTrue(np.array_equal(db_custom.neuron_names, neuron_names_custom))
        self.assertIsNone(db_custom.pred_model)

    def test_exclude_neurons(self):
        # Test the exclude_neurons function
        # Include cases with valid and invalid neuron names

        db = Database(neuron_traces=[[1, 2], [4, 5]], behavior=[0, 1], neuron_names=['neuron1', 'neuron2'])

        # Exclude an existing neuron
        db.exclude_neurons(['neuron1'])
        self.assertTrue(np.array_equal(db.neuron_traces, np.array([[4, 5]])))
        self.assertTrue(np.array_equal(db.neuron_names, np.array(['neuron2'])))

        # Exclude a non-existing neuron
        db.exclude_neurons(['nonexistent_neuron'])
        self.assertTrue(np.array_equal(db.neuron_traces, np.array([[4, 5]])))
        self.assertTrue(np.array_equal(db.neuron_names, np.array(['neuron2'])))

    def test_createVisualizer(self):
        # Test the createVisualizer function
        # Include cases with and without specifying a mapping

        db = Database(neuron_traces=[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                     [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
                                     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
                      behavior=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                      neuron_names=['neuron1', 'neuron2', 'neuron3', 'neuron4'],
                      fps=1)

        # Without mapping
        visualizer_no_mapping = db.createVisualizer(window=3, epochs=20)
        self.assertIsNotNone(visualizer_no_mapping)
        self.assertIsNotNone(visualizer_no_mapping.mapping)
        self.assertIsNotNone(visualizer_no_mapping.tau_model)

        # With PCA mapping
        from sklearn.decomposition import PCA
        mapping_pca = PCA(n_components=2)
        visualizer_with_mapping = db.createVisualizer(mapping=mapping_pca)
        self.assertIsNotNone(visualizer_with_mapping)
        self.assertEqual(visualizer_with_mapping.mapping, mapping_pca)
        self.assertIsNone(visualizer_with_mapping.tau_model)

    # Similar tests can be created for the remaining functions

    def tearDown(self):
        # Clean up any resources or configurations used in the tests
        pass


if __name__ == '__main__':
    unittest.main()
