import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


COLLADA_NS = {"c": "http://www.collada.org/2005/11/COLLADASchema"}
DEFAULT_SCALE = 0.01


def _read_float_array(source):
    float_array = source.find("c:float_array", COLLADA_NS)
    if float_array is None or not float_array.text:
        return []
    return [float(value) for value in float_array.text.split()]


def _source_arrays(root):
    arrays = {}
    for source in root.findall(".//c:source", COLLADA_NS):
        source_id = source.attrib.get("id")
        if source_id:
            arrays[source_id] = _read_float_array(source)
    return arrays


def _vertices_sources(root):
    vertices = {}
    for vertices_node in root.findall(".//c:vertices", COLLADA_NS):
        vertices_id = vertices_node.attrib.get("id")
        if not vertices_id:
            continue
        for input_node in vertices_node.findall("c:input", COLLADA_NS):
            if input_node.attrib.get("semantic") == "POSITION":
                vertices[vertices_id] = input_node.attrib["source"].lstrip("#")
    return vertices


def _material_for_polylist(polylist):
    material = polylist.attrib.get("material")
    if material:
        return material.replace("-material", "")
    return "nao_texture"


def convert_dae_to_obj(dae_path, obj_path, material_name, texture_relpath, scale):
    root = ET.parse(dae_path).getroot()
    arrays = _source_arrays(root)
    vertices = _vertices_sources(root)

    obj_path.parent.mkdir(parents=True, exist_ok=True)
    mtl_path = obj_path.with_suffix(".mtl")

    obj_lines = [
        f"mtllib {mtl_path.name}",
        f"o {dae_path.stem}",
        f"usemtl {material_name}",
    ]

    vertex_offset = 1
    texcoord_offset = 1

    for mesh in root.findall(".//c:geometry/c:mesh", COLLADA_NS):
        for polylist in mesh.findall("c:polylist", COLLADA_NS):
            inputs = []
            max_offset = 0
            for input_node in polylist.findall("c:input", COLLADA_NS):
                semantic = input_node.attrib["semantic"]
                source = input_node.attrib["source"].lstrip("#")
                offset = int(input_node.attrib.get("offset", "0"))
                if semantic == "VERTEX":
                    source = vertices[source]
                inputs.append((semantic, source, offset))
                max_offset = max(max_offset, offset)

            position_source = next(source for semantic, source, _ in inputs if semantic == "VERTEX")
            texcoord_source = next(
                (source for semantic, source, _ in inputs if semantic == "TEXCOORD"),
                None,
            )
            position_values = arrays[position_source]
            texcoord_values = arrays.get(texcoord_source, [])

            for idx in range(0, len(position_values), 3):
                x, y, z = position_values[idx : idx + 3]
                obj_lines.append(f"v {x * scale:.9g} {y * scale:.9g} {z * scale:.9g}")

            if texcoord_values:
                for idx in range(0, len(texcoord_values), 2):
                    u, v = texcoord_values[idx : idx + 2]
                    obj_lines.append(f"vt {u:.9g} {1.0 - v:.9g}")

            obj_lines.append(f"usemtl {_material_for_polylist(polylist)}")

            vcount_node = polylist.find("c:vcount", COLLADA_NS)
            p_node = polylist.find("c:p", COLLADA_NS)
            if vcount_node is None or p_node is None or not p_node.text:
                continue

            stride = max_offset + 1
            counts = [int(value) for value in vcount_node.text.split()]
            packed = [int(value) for value in p_node.text.split()]
            cursor = 0
            input_by_offset = {offset: semantic for semantic, _, offset in inputs}

            for count in counts:
                polygon = []
                for _ in range(count):
                    record = packed[cursor : cursor + stride]
                    cursor += stride
                    vertex_index = None
                    texcoord_index = None
                    for offset, semantic in input_by_offset.items():
                        if semantic == "VERTEX":
                            vertex_index = record[offset] + vertex_offset
                        elif semantic == "TEXCOORD" and texcoord_values:
                            texcoord_index = record[offset] + texcoord_offset
                    polygon.append((vertex_index, texcoord_index))

                for idx in range(1, len(polygon) - 1):
                    tri = (polygon[0], polygon[idx], polygon[idx + 1])
                    face = []
                    for vertex_index, texcoord_index in tri:
                        if texcoord_index is None:
                            face.append(str(vertex_index))
                        else:
                            face.append(f"{vertex_index}/{texcoord_index}")
                    obj_lines.append("f " + " ".join(face))

            vertex_offset += len(position_values) // 3
            texcoord_offset += len(texcoord_values) // 2

    material_names = sorted(
        {
            line.removeprefix("usemtl ")
            for line in obj_lines
            if line.startswith("usemtl ")
        }
    )
    mtl_lines = []
    for name in material_names:
        mtl_lines.extend(
            [
                f"newmtl {name}",
                "Ka 0.2 0.2 0.2",
                "Kd 1.0 1.0 1.0",
                "Ks 0.35 0.35 0.35",
                "Ns 50",
                f"map_Kd {texture_relpath}",
                "",
            ]
        )

    obj_path.write_text("\n".join(obj_lines) + "\n")
    mtl_path.write_text("\n".join(mtl_lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_dir", type=Path, default=Path("assets/nao/meshes/V40"))
    parser.add_argument("--output_dir", type=Path, default=Path("assets/nao/meshes/V40_obj"))
    parser.add_argument("--texture_relpath", default="../../texture/textureNAO.png")
    parser.add_argument("--scale", type=float, default=DEFAULT_SCALE)
    args = parser.parse_args()

    for dae_path in sorted(args.mesh_dir.glob("*.dae")):
        obj_path = args.output_dir / f"{dae_path.stem}.obj"
        convert_dae_to_obj(
            dae_path,
            obj_path,
            material_name="nao_texture",
            texture_relpath=args.texture_relpath,
            scale=args.scale,
        )
        print(f"wrote {obj_path}")


if __name__ == "__main__":
    main()
