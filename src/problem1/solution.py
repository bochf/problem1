"""
Solution for Problem 1
"""

import argparse
from pathlib import Path

from matplotlib import pyplot as plt

from problem1.route_planing import RoutePlanning

DEFAULT_MAP = Path(__file__).parent / "map.png"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Route Planning")
    parser.add_argument(
        "--map", type=str, default=str(DEFAULT_MAP), help="Path to the map image"
    )
    parser.add_argument(
        "--route_type",
        type=str,
        choices=[RoutePlanning.ROUTE_TYPE_GRID, RoutePlanning.ROUTE_TYPE_PRM],
        default=RoutePlanning.ROUTE_TYPE_GRID,
        help="Type of route planning algorithm to use (grid or prm)",
    )
    parser.add_argument(
        "--samples", type=int, default=2500, help="Number of PRM samples"
    )
    parser.add_argument(
        "--d_max", type=float, default=75.0, help="Maximum PRM connection radius"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for PRM sampling"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    # Create a RoutePlanning instance with the map image
    route_planner = RoutePlanning(args.map, seed=args.seed)

    # Find the shortest path from start to goal
    path, cost = route_planner.shortest_path(
        args.route_type, num_samples=args.samples, d_max=args.d_max
    )

    if args.route_type == RoutePlanning.ROUTE_TYPE_PRM:
        route_planner.plot_roadmap(
            title=f"PRM (N={args.samples}, d_max={args.d_max:g})"
        )

    # Print the results
    if path:
        print(f"Shortest path found with cost {cost:.2f} ({len(path)} vertices)")
        title = (
            "Grid A* path"
            if args.route_type == RoutePlanning.ROUTE_TYPE_GRID
            else "PRM path"
        )
        route_planner.print_map(path, title=f"{title} (length {cost:.2f})")
    else:
        print("No path found.")

    plt.show()
