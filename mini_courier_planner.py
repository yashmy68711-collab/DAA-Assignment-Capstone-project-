"""
MINI-COURIER PLANNER
Design & Analysis of Algorithms (ETCCDA302)
K.R. Mangalam University - Capstone Assignment

Algorithms used:
1. Merge Sort       : prioritize parcels by priority score (descending).
2. 0/1 Knapsack DP  : choose the best parcels within the vehicle weight capacity.
3. Dijkstra        : shortest paths between delivery locations in a weighted graph.
4. Nearest Neighbor: approximate a delivery tour (TSP) using shortest-path distances.
5. Performance     : measure algorithm runtimes on different input sizes.

Run:
    python mini_courier_planner.py

No third-party libraries are required.
"""

import heapq
import random
import time
import csv
import json
from pathlib import Path

DATA_FILE = Path(__file__).with_name("sample_parcels.json")

def load_input_data():
    with DATA_FILE.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return (
        data["depot"],
        data["vehicle_capacity"],
        data["parcels"],
        [tuple(road) for road in data["roads"]],
    )

DEPOT, VEHICLE_CAPACITY, PARCELS, ROADS = load_input_data()

def build_graph(roads):
    """Create an adjacency-list representation of the road network."""
    graph = {}
    for start, end, distance in roads:
        graph.setdefault(start, []).append((end, distance))
        graph.setdefault(end, []).append((start, distance))
    return graph

def parcel_score(parcel):
    """
    Larger priority and earlier deadlines should rank higher.
    The deadline bonus is a simple scoring rule for this prototype.
    """
    return parcel["priority"] * 100 - parcel["deadline"]


def merge_sort_parcels(parcels):
    """Merge sort parcels by score in descending order."""
    if len(parcels) <= 1:
        return parcels[:]

    middle = len(parcels) // 2
    left = merge_sort_parcels(parcels[:middle])
    right = merge_sort_parcels(parcels[middle:])

    result = []
    i = j = 0

    while i < len(left) and j < len(right):
        if parcel_score(left[i]) >= parcel_score(right[j]):
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])
    return result

def knapsack_parcels(parcels, capacity):
    """
    0/1 Knapsack using dynamic programming.
    Returns the selected parcels and their total priority score.
    Each parcel is either selected once or not selected.
    """
    n = len(parcels)
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]

    for i in range(1, n + 1):
        weight = parcels[i - 1]["weight"]
        value = parcel_score(parcels[i - 1])

        for current_capacity in range(capacity + 1):
            dp[i][current_capacity] = dp[i - 1][current_capacity]

            if weight <= current_capacity:
                include_value = value + dp[i - 1][current_capacity - weight]
                dp[i][current_capacity] = max(
                    dp[i][current_capacity], include_value
                )

    selected = []
    remaining_capacity = capacity

    for i in range(n, 0, -1):
        if dp[i][remaining_capacity] != dp[i - 1][remaining_capacity]:
            parcel = parcels[i - 1]
            selected.append(parcel)
            remaining_capacity -= parcel["weight"]

    selected.reverse()
    return selected, dp[n][capacity], dp

def dijkstra(graph, source):
    """Find shortest distances and previous nodes from source."""
    distances = {node: float("inf") for node in graph}
    previous = {node: None for node in graph}
    distances[source] = 0

    priority_queue = [(0, source)]

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        if current_distance > distances[current_node]:
            continue

        for neighbor, edge_distance in graph[current_node]:
            new_distance = current_distance + edge_distance

            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                previous[neighbor] = current_node
                heapq.heappush(priority_queue, (new_distance, neighbor))

    return distances, previous


def reconstruct_path(previous, source, destination):
    """Reconstruct a shortest path using Dijkstra's previous-node map."""
    path = []
    current = destination

    while current is not None:
        path.append(current)
        current = previous[current]

    path.reverse()
    if not path or path[0] != source:
        return []
    return path

def nearest_neighbor_route(graph, start, destinations):
    """
    Approximate a TSP tour:
    repeatedly visit the nearest unvisited destination, then return to depot.
    Distances between stops are shortest-road distances from Dijkstra.
    """
    remaining = set(destinations)
    remaining.discard(start)

    current = start
    route = [start]
    total_distance = 0
    route_legs = []

    while remaining:
        distances, previous = dijkstra(graph, current)
        reachable = [place for place in remaining if distances.get(place, float("inf")) < float("inf")]

        if not reachable:
            raise ValueError("Some delivery locations cannot be reached from the current location.")

        next_place = min(reachable, key=lambda place: distances[place])
        leg_distance = distances[next_place]
        leg_path = reconstruct_path(previous, current, next_place)

        route_legs.append((current, next_place, leg_path, leg_distance))
        total_distance += leg_distance
        route.append(next_place)
        remaining.remove(next_place)
        current = next_place

    if current != start:
        distances, previous = dijkstra(graph, current)
        if distances.get(start, float("inf")) == float("inf"):
            raise ValueError("The depot cannot be reached from the final delivery location.")

        leg_distance = distances[start]
        leg_path = reconstruct_path(previous, current, start)
        route_legs.append((current, start, leg_path, leg_distance))
        total_distance += leg_distance
        route.append(start)

    return route, total_distance, route_legs

def measure_runtime(function, *args, repeats=3):
    """Return the average runtime in milliseconds."""
    total = 0.0
    for _ in range(repeats):
        start_time = time.perf_counter()
        function(*args)
        total += time.perf_counter() - start_time
    return (total / repeats) * 1000


def generate_test_parcels(count, seed=42):
    """Create repeatable synthetic parcels for performance testing."""
    rng = random.Random(seed + count)
    parcels = []
    for index in range(count):
        parcels.append({
            "id": f"T{index + 1}",
            "destination": "A",
            "weight": rng.randint(1, 8),
            "priority": rng.randint(1, 10),
            "deadline": rng.randint(8, 20),
        })
    return parcels


def run_performance_tests():
    """
    Compare sorting and knapsack runtimes for increasing input sizes.
    Dijkstra and nearest-neighbor are measured on generated road graphs.
    Results are saved to performance_results.csv.
    """
    sizes = [10, 25, 50, 100, 200]
    results = []

    for size in sizes:
        test_parcels = generate_test_parcels(size)
        capacity = 30

        sort_ms = measure_runtime(merge_sort_parcels, test_parcels)
        knapsack_ms = measure_runtime(knapsack_parcels, test_parcels, capacity)

        test_graph = {}
        test_nodes = [f"N{i}" for i in range(10)]
        for i in range(len(test_nodes) - 1):
            a, b = test_nodes[i], test_nodes[i + 1]
            test_graph.setdefault(a, []).append((b, 1))
            test_graph.setdefault(b, []).append((a, 1))

        dijkstra_ms = measure_runtime(dijkstra, test_graph, test_nodes[0])
        route_stops = test_nodes[1: min(6, len(test_nodes))]
        tsp_ms = measure_runtime(nearest_neighbor_route, test_graph, test_nodes[0], route_stops)

        results.append({
            "input_size": size,
            "merge_sort_ms": round(sort_ms, 6),
            "knapsack_dp_ms": round(knapsack_ms, 6),
            "dijkstra_ms": round(dijkstra_ms, 6),
            "nearest_neighbor_ms": round(tsp_ms, 6),
        })

    output_file = Path("performance_results.csv")
    with output_file.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = list(results[0].keys())
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("\nPERFORMANCE TEST RESULTS (average milliseconds)")
    print("-" * 78)
    print(f"{'Input':>8} {'Merge Sort':>14} {'Knapsack DP':>14} {'Dijkstra':>14} {'Nearest Neighbor':>18}")
    for row in results:
        print(
            f"{row['input_size']:>8} "
            f"{row['merge_sort_ms']:>14.6f} "
            f"{row['knapsack_dp_ms']:>14.6f} "
            f"{row['dijkstra_ms']:>14.6f} "
            f"{row['nearest_neighbor_ms']:>18.6f}"
        )
    print(f"\nCSV report saved as: {output_file.resolve()}")
    print("Note: route algorithms use a fixed 10-node test graph; parcel count varies.")

def print_parcels(title, parcels):
    print(f"\n{title}")
    print("-" * 75)
    if not parcels:
        print("No parcels.")
        return

    print(f"{'ID':<8}{'Destination':<15}{'Weight(kg)':<15}{'Priority':<12}{'Deadline':<10}")
    for parcel in parcels:
        print(
            f"{parcel['id']:<8}"
            f"{parcel['destination']:<15}"
            f"{parcel['weight']:<15}"
            f"{parcel['priority']:<12}"
            f"{parcel['deadline']:<10}"
        )


def main():
    print("=" * 75)
    print("                 MINI-COURIER PLANNER")
    print("          Design & Analysis of Algorithms (ETCCDA302)")
    print("=" * 75)

    graph = build_graph(ROADS)

    sorted_parcels = merge_sort_parcels(PARCELS)
    print_parcels("1. PARCELS AFTER MERGE SORT (highest score first)", sorted_parcels)

    selected_parcels, best_score, dp_table = knapsack_parcels(
        sorted_parcels, VEHICLE_CAPACITY
    )
    selected_weight = sum(parcel["weight"] for parcel in selected_parcels)

    print_parcels("2. SELECTED PARCELS (0/1 KNAPSACK)", selected_parcels)
    print(f"\nVehicle capacity : {VEHICLE_CAPACITY} kg")
    print(f"Selected weight  : {selected_weight} kg")
    print(f"Total score      : {best_score}")

    print("\nKnapsack DP table (rows = parcels considered, columns = capacity):")
    for row in dp_table:
        print(" ".join(f"{value:5}" for value in row))

    destinations = sorted({parcel["destination"] for parcel in selected_parcels})
    if not destinations:
        print("\nNo parcels selected; no delivery route is required.")
    else:
        print("\n3. SHORTEST PATHS FROM DEPOT (DIJKSTRA)")
        print("-" * 75)
        distances, previous = dijkstra(graph, DEPOT)

        for destination in destinations:
            path = reconstruct_path(previous, DEPOT, destination)
            print(
                f"{DEPOT} -> {destination}: "
                f"{' -> '.join(path)} | Distance = {distances[destination]} km"
            )

        route, route_distance, legs = nearest_neighbor_route(
            graph, DEPOT, destinations
        )

        print("\n4. FINAL DELIVERY ROUTE (NEAREST-NEIGHBOR TSP APPROXIMATION)")
        print("-" * 75)
        print(" -> ".join(route))
        print(f"Total route distance: {route_distance} km")

        print("\nRoute leg details:")
        for start, end, path, distance in legs:
            print(f"{start} to {end}: {' -> '.join(path)} ({distance} km)")

    print("\n5. PERFORMANCE TESTING")
    run_performance_tests()

    print("\n6. COMPLEXITY SUMMARY")
    print("-" * 75)
    print("Merge Sort       : Time O(n log n), Space O(n)")
    print("0/1 Knapsack DP  : Time O(nW), Space O(nW), W = capacity")
    print("Dijkstra         : O((V + E) log V) with an adjacency list and heap")
    print("Nearest Neighbor : O(k^2) distance-selection steps, plus shortest-path runs")
    print("Graph storage    : O(V + E)")
    print("\nProgram completed successfully.")


if __name__ == "__main__":
    main()
