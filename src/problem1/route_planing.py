from PIL import Image
import networkx as nx
from networkx import Graph
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

# A grid cell (row, col). Plain tuples are hashable, so they are used directly as graph nodes.
Cell = tuple[int, int]


def _convert_image_to_grid(image: str) -> np.ndarray:
    """
    Get the data from the image

    :param image: The path to the image
    :return: The 2d array of the map, where 1 represents free cells and 0 represents obstacles
    """

    img = Image.open(image)
    img = img.convert("L")  # Convert to grayscale
    data = (np.array(img) > 127).astype(int)
    return data


def _euclidean_distance(cell1: Cell, cell2: Cell) -> float:
    """
    d(v1, v2): the Euclidean distance between two cell centers.

    Also used directly as the A* heuristic, since NetworkX calls heuristic(node, goal).

    :param cell1: The first cell
    :param cell2: The second cell
    :return: The Euclidean distance between the two cells
    """
    return float(np.hypot(cell1[0] - cell2[0], cell1[1] - cell2[1]))


class RoutePlanning:
    ROUTE_TYPE_GRID = "grid"
    ROUTE_TYPE_PRM = "prm"

    def __init__(self, image: str, seed: int | None = None):
        # Convert the image to a 2D array of free and obstacle cells
        self._map = _convert_image_to_grid(image)
        # Array of shape (K, 2): one (row, col) per free cell, used for sampling
        self._free_cells = np.argwhere(self._map == 1)
        if len(self._free_cells) == 0:
            raise ValueError("No free cells available in the map.")

        # Seeded random generator so PRM results are reproducible
        self._rng = np.random.default_rng(seed)

        self._start: Cell = (635, 140)
        self._goal: Cell = (350, 400)
        for name, cell in (("start", self._start), ("goal", self._goal)):
            if not self._is_free(cell):
                raise ValueError(f"The {name} cell {cell} is not a free cell.")

        # The graph used by the last call to shortest_path (for plotting)
        self.graph: Graph | None = None

    # ------------------------------------------------------------------ #
    # Basic grid helpers
    # ------------------------------------------------------------------ #
    def _is_free(self, cell: Cell) -> bool:
        row, col = cell
        max_rows, max_cols = self._map.shape
        return 0 <= row < max_rows and 0 <= col < max_cols and self._map[row, col] == 1

    def neighbors(self, cell: Cell) -> list[Cell]:
        """
        N(v): get the free 8-connected neighbors of a cell, respecting the map boundaries.

        :param cell: The cell to get the neighbors of
        :return: A list of the free 8-connected neighbors of the cell
        """
        r, c = cell
        neighbors = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                neighbor = (r + dr, c + dc)
                if self._is_free(neighbor):
                    neighbors.append(neighbor)
        return neighbors

    # ------------------------------------------------------------------ #
    # Part (a): grid graph
    # ------------------------------------------------------------------ #
    def _build_grid_graph(self) -> Graph:
        """
        Build a graph from the map data, where each free cell is a node and edges exist between 8-connected free cells.

        :return: A NetworkX graph representing the free cells and their connections
        """
        graph = Graph()
        for r, c in self._free_cells:
            cell = (int(r), int(c))
            graph.add_node(cell)
            for neighbor in self.neighbors(cell):
                if not graph.has_edge(cell, neighbor):
                    graph.add_edge(
                        cell, neighbor, weight=_euclidean_distance(cell, neighbor)
                    )
        return graph

    # ------------------------------------------------------------------ #
    # Part (b): PRM
    # ------------------------------------------------------------------ #
    def sample_cell(self) -> Cell:
        """
        Sample a free cell uniformly at random from the map.

        :return: A randomly selected free cell
        """
        row, col = self._free_cells[self._rng.integers(len(self._free_cells))]
        return (int(row), int(col))

    def is_reachable(self, cell1: Cell, cell2: Cell) -> bool:
        """
        Straight-line local planner: check whether the segment joining the centers of cell1 and cell2
        is obstacle-free, i.e. it does not intersect any occupied cell.

        Walks every grid cell the segment passes through (exact integer arithmetic). Cell (r, c) is the
        unit square centered at (r, c), so the segment crosses a row boundary at t = (i + 0.5) / |dr| and
        a column boundary at t = (j + 0.5) / |dc|. If it passes exactly through a cell corner, both cells
        touching that corner are checked as well, so the line can't slip between diagonal obstacles.

        :param cell1: The first cell
        :param cell2: The second cell
        :return: True if the straight segment between the two cells is obstacle-free
        """
        r, c = cell1
        dr, dc = cell2[0] - r, cell2[1] - c
        step_r, step_c = int(np.sign(dr)), int(np.sign(dc))
        n_r, n_c = abs(dr), abs(dc)
        i = j = 0  # number of row / column boundaries crossed so far

        if self._map[r, c] == 0:
            return False
        while i < n_r or j < n_c:
            # Compare the next crossing times (2i+1)/(2 n_r) and (2j+1)/(2 n_c) without division
            t_row = (2 * i + 1) * n_c
            t_col = (2 * j + 1) * n_r
            if j >= n_c or (i < n_r and t_row < t_col):
                r += step_r
                i += 1
            elif i >= n_r or t_col < t_row:
                c += step_c
                j += 1
            else:  # passes exactly through a corner
                if self._map[r + step_r, c] == 0 or self._map[r, c + step_c] == 0:
                    return False
                r += step_r
                c += step_c
                i += 1
                j += 1
            if self._map[r, c] == 0:
                return False
        return True

    def _add_prm_vertex(self, graph: Graph, new_cell: Cell, d_max: float) -> None:
        """
        Algorithm 2 (AddVertex): add new_cell to the roadmap and connect it to every existing vertex
        within d_max that is reachable with the straight-line local planner.
        """
        existing = list(graph.nodes)
        graph.add_node(new_cell)
        if not existing:
            return

        # Vectorized distance filter, then the (more expensive) collision check on close vertices only
        points = np.array(existing)
        distances = np.hypot(points[:, 0] - new_cell[0], points[:, 1] - new_cell[1])
        for idx in np.flatnonzero((distances <= d_max) & (distances > 0)):
            other = existing[idx]
            if self.is_reachable(new_cell, other):
                graph.add_edge(new_cell, other, weight=float(distances[idx]))

    def _build_prm_graph(self, num_samples: int, d_max: float) -> Graph:
        """
        Algorithm 1 (ConstructPRM): build a Probabilistic Roadmap with num_samples distinct free cells.

        :param num_samples: The number of random free cells to sample
        :param d_max: The maximum local search radius
        :return: A NetworkX graph representing the PRM
        """
        if len(self._free_cells) < num_samples:
            raise ValueError(
                f"Not enough free cells to sample {num_samples} cells. "
                f"Available free cells: {len(self._free_cells)}"
            )

        prm_graph = Graph()
        while prm_graph.number_of_nodes() < num_samples:
            cell = self.sample_cell()
            if cell in prm_graph:  # skip duplicate samples
                continue
            self._add_prm_vertex(prm_graph, cell, d_max)
        return prm_graph

    # ------------------------------------------------------------------ #
    # Planning
    # ------------------------------------------------------------------ #
    def shortest_path(
        self,
        route_type: str,
        num_samples: int = 2500,
        d_max: float = 75.0,
        max_extra_samples: int = 10000,
    ) -> tuple[list[Cell], float]:
        """
        Find the shortest path from the start cell to the goal cell using NetworkX's A* search,
        with Euclidean edge costs and the Euclidean distance to the goal as the heuristic.

        :param route_type: ROUTE_TYPE_GRID or ROUTE_TYPE_PRM
        :param num_samples: Number of PRM samples (PRM only)
        :param d_max: Maximum PRM connection radius (PRM only)
        :param max_extra_samples: Upper bound on extra PRM samples added if start and goal are disconnected
        :return: A list of cells representing the shortest path from start to goal and the total cost of the path
        """
        start, goal = self._start, self._goal

        if route_type == self.ROUTE_TYPE_GRID:
            graph = self._build_grid_graph()
        elif route_type == self.ROUTE_TYPE_PRM:
            graph = self._build_prm_graph(num_samples, d_max)
            # (b)(v): connect start and goal with the same distance and collision rules
            self._add_prm_vertex(graph, start, d_max)
            self._add_prm_vertex(graph, goal, d_max)
            # Keep sampling until start and goal are in the same connected component
            extra = 0
            while not nx.has_path(graph, start, goal) and extra < max_extra_samples:
                cell = self.sample_cell()
                if cell not in graph:
                    self._add_prm_vertex(graph, cell, d_max)
                    extra += 1
            if extra:
                print(f"Added {extra} extra samples to connect start and goal.")
        else:
            raise ValueError(f"Unknown route type: {route_type}")

        self.graph = graph
        try:
            path = nx.astar_path(
                graph, start, goal, heuristic=_euclidean_distance, weight="weight"
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
            print(f"Error finding path: {e}")
            return [], float("inf")

        return path, nx.path_weight(graph, path, weight="weight")

    # ------------------------------------------------------------------ #
    # Plotting (cells are (row, col), so x = col and y = row)
    # ------------------------------------------------------------------ #
    def _plot_base(self, title: str):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(self._map, cmap="gray")  # 0 = black (occupied), 1 = white (free)
        ax.set_title(title)
        return fig, ax

    def _plot_start_goal(self, ax) -> None:
        ax.plot(self._start[1], self._start[0], "go", markersize=8, label="start")
        ax.plot(self._goal[1], self._goal[0], "bo", markersize=8, label="goal")
        ax.legend()

    def plot_roadmap(
        self, graph: Graph | None = None, title: str = "Probabilistic roadmap"
    ):
        """
        Plot the roadmap (vertices and edges) overlaid on the occupancy grid map.
        """
        graph = graph if graph is not None else self.graph
        if graph is None:
            raise ValueError("No graph to plot; call shortest_path first.")
        _, ax = self._plot_base(title)
        segments = [[(u[1], u[0]), (v[1], v[0])] for u, v in graph.edges]
        ax.add_collection(
            LineCollection(segments, colors="tab:blue", linewidths=0.4, alpha=0.6)
        )
        nodes = np.array(list(graph.nodes))
        ax.scatter(nodes[:, 1], nodes[:, 0], s=2, c="tab:orange", zorder=3)
        self._plot_start_goal(ax)

    def print_map(self, path: list[Cell] | None = None, title: str = "A* path"):
        """
        Plot the map with the path overlaid: free cells white, obstacles black, path red.

        :param path: A list of cells representing the path to overlay on the map
        :param title: The figure title
        """
        _, ax = self._plot_base(title)
        if path:
            rows, cols = zip(*path)
            ax.plot(cols, rows, "r-", linewidth=2)
        self._plot_start_goal(ax)
