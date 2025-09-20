    # Example candidate mapping based on your earlier tests:
    hex_region_mapping = {
        74: [ 16, 41, 42, 66, 73],
        75: [ 16, 17, 18, 19, 20],
        76: [ 16, 17, 18, 19, 20],
        77: [ 20, 56, 71, 72, 73],
        78: [ 16, 41, 42, 66, 73]
    }

    # List of pentagon vertex IDs (for example, as output previously)
    pentagon_vertex_ids = [74, 75, 76, 77, 78]

    # Call the connection function from your graph instance:
    new_edges = g1.connect_pentagon_vertices(pentagon_vertex_ids, hex_region_mapping)





    print("New connection edges:", new_edges)

    new_connections = {
        74: 66,  # distance 0.0877
        75: 17,  # distance 0.0722
        76: 19,  # distance 0.1178
        77: 72,  # distance 0.1192
        78: 42   # distance 0.0999
    }

    # Create a new figure and axis.
    fig, ax = plt.subplots(figsize=(8, 6))


    new_connections = {
        74: 66,  # distance 0.0877
        75: 17,  # distance 0.0722
        76: 19,  # distance 0.1178
        77: 72,  # distance 0.1192
        78: 42   # distance 0.0999
    }

    # Create a new figure and axis.
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the existing structure (using region boundaries stored in g.region_coors).
    for coors in g1.region_coors.values():
        # Ensure each region has at least two vertices.
        if len(coors) < 2:
            continue
        for i in range(len(coors)):
            cur = coors[i]
            nxt = coors[i+1] if i < len(coors) - 1 else coors[0]
            ax.plot([cur[0], nxt[0]], [cur[1], nxt[1]], color='k', linewidth=1)

    # Now overlay the new connection edges.
    for pent_v, hex_v in new_connections.items():
        P = g1.vertices[pent_v]  # coordinate of the pentagon vertex
        H = g1.vertices[hex_v]   # coordinate of the hexagon vertex
        # Compute Euclidean distance manually (compatible with older Python versions)
        distance = math.sqrt((P[0] - H[0])**2 + (P[1] - H[1])**2)
        
        # Plot the new edge in red, dashed line.
        ax.plot([P[0], H[0]], [P[1], H[1]], color='red', linewidth=2, linestyle='--')
        
        # Annotate the edge at its midpoint with the distance.
        mid = ((P[0] + H[0]) / 2, (P[1] + H[1]) / 2)
        ax.text(mid[0], mid[1], f'{distance:.4f}', color='blue', fontsize=9)

    # Optionally, draw the pentagon vertices in a distinct marker.
    for v in new_connections.keys():
        x, y = g1.vertices[v]
        ax.plot(x, y, 'bo', markersize=6)

    # Optionally, draw hexagon candidate vertices in a distinct marker.
    for v in new_connections.values():
        x, y = g1.vertices[v]
        ax.plot(x, y, 'go', markersize=6)

    # Add some axis formatting.
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('New Connected Edges from Pentagon to Hexagon Vertices')
    plt.savefig('plot_with_new_edges.png', dpi=300)


    g1.snap() 