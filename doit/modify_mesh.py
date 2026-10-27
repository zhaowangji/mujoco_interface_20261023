with open("nx_keti.txt", "w", encoding="utf-8") as f:
    for i in range(765):
        f.write(
            f'    <mesh name="nx_pas_collision_{i}" '
            f'file="nx_pas_better/nx_pas_collision_{i}.obj" '
            f'scale="0.001 0.001 0.001"/>\n'
        )