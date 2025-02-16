# Self-Organizing Map (SOM) Implementation

## Project Overview
This project implements a Self-Organizing Map (SOM), also known as a Kohonen Network, from scratch using Python. SOMs are a type of artificial neural network that can be used for dimensionality reduction, visualization, and clustering of high-dimensional data.

## Features
- Custom SOM implementation with numpy
- Visualization tools for understanding the network
- U-Matrix visualization for cluster analysis
- Component planes visualization
- Interactive training process with progress bar
- Flexible map size and training parameters

## Prerequisites
```
numpy
matplotlib
scikit-learn
torch
tqdm
```

## Technical Details

### SOM Architecture
The implementation includes:
- 2D grid of neurons with random initial weights
- Gaussian neighborhood function
- Learning rate decay over time
- Best Matching Unit (BMU) calculation
- U-Matrix generation for cluster visualization

### Key Components

1. **Initialization**
```python
def __init__(self, input_dim, map_size=(10, 10), learning_rate=0.1, sigma=1.0):
    self.weights = np.random.random((map_size[0], map_size[1], input_dim))
```

2. **BMU Finding**
```python
def find_bmu(self, input_vector):
    distances = np.sum((self.weights - input_vector) ** 2, axis=2)
    return np.unravel_index(np.argmin(distances), self.map_size)
```

3. **Training Process**
- Iterative weight updates
- Learning rate decay
- Neighborhood function application
- Progress tracking with tqdm

### Visualizations
The project provides several visualization methods:
1. Original data distribution
2. U-Matrix for cluster identification
3. SOM grid with connections
4. Component planes for feature analysis

## Usage

### Basic Usage
```python
# Create SOM instance
som = SelfOrganizingMap(input_dim=2, map_size=(20, 20))

# Train the network
som.train(data, epochs=50)

# Generate U-Matrix
u_matrix = som.get_u_matrix()
```

### Customizing Parameters
```python
som = SelfOrganizingMap(
    input_dim=2,
    map_size=(30, 30),
    learning_rate=0.05,
    sigma=2.0
)
```

## Model Parameters

- `input_dim`: Dimensionality of input data
- `map_size`: Tuple defining the grid size (height, width)
- `learning_rate`: Initial learning rate for weight updates
- `sigma`: Initial radius of the neighborhood function

## Applications

1. **Data Visualization**
   - Reducing high-dimensional data to 2D representation
   - Visualizing cluster structures

2. **Pattern Recognition**
   - Identifying clusters in complex datasets
   - Feature mapping and analysis

3. **Dimensionality Reduction**
   - Alternative to PCA or t-SNE
   - Topology-preserving mapping

## Future Improvements

1. **Performance Optimization**
   - Vectorized implementation of weight updates
   - GPU acceleration support

2. **Additional Features**
   - More neighborhood functions
   - Advanced visualization tools
   - Automatic parameter optimization

3. **Analysis Tools**
   - Quantization error calculation
   - Topographic error measurement
   - Cluster validation metrics

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License.

## References
- Kohonen, T. (1990). The self-organizing map. Proceedings of the IEEE
- Vesanto, J. (1999). SOM-based data visualization methods