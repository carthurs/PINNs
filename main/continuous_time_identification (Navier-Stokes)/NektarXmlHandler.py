import xml.etree.ElementTree as ET
import itertools
import collections


def iterify(to_be_made_iterable):
    if (isinstance(to_be_made_iterable, collections.Iterable)):
        return to_be_made_iterable
    else:
        return [to_be_made_iterable]

class NektarXmlHandler(object):
    def __init__(self, xml_file_name_in):
        self.xml_file_name = xml_file_name_in
        self.xml_tree = ET.parse(self.xml_file_name)

    # Returns the first occurrence of a child of the parent, matching child_tag
    def get_child_tag(self, parent, child_tag):
        for child in parent:
            if child.tag == child_tag:
                return child

    def _find_composites(self):
        return self.xml_tree.getroot().find('GEOMETRY').find('COMPOSITE')

    def _find_edges(self):
        return self.xml_tree.getroot().find('GEOMETRY').find('EDGE')

    def _find_vertices(self):
        return self.xml_tree.getroot().find('GEOMETRY').find('VERTEX')

    def find_composite_by_id(self, id):
        id = str(id)
        composites_parent_node = self._find_composites()
        for composite in composites_parent_node:
            if composite.attrib['ID'] == id:
                IDs = [int(ii) for ii in composite.text[3:-2].split(',')]
                return IDs

    def get_nodes_of_edge(self, id):
        id = str(id)
        edges_parent_node = self._find_edges()
        for edge in edges_parent_node:
            if edge.attrib['ID'] == id:
                nodes_of_edge = [int(ii) for ii in edge.text.split()]
                return nodes_of_edge

    def get_coordinates_of_node(self, node_id):
        node_id = str(node_id)
        vertices_parent_node = self._find_vertices()
        for vertex in vertices_parent_node:
            if vertex.attrib['ID'] == node_id:
                coordinates = [float(ii) for ii in vertex.text.split()]
                return coordinates

    # Can take either a composite ID, or a list of composite IDs (due to the iterify function)
    def get_nodes_in_composite(self, composite_ids):
        node_pairs_in_edges_in_composites = []
        for composite_id in iterify(composite_ids):
            edges_in_composite = self.find_composite_by_id(composite_id)
            print(composite_id, edges_in_composite)
            node_pairs_in_edges_in_composite = [self.get_nodes_of_edge(edge_id) for edge_id in edges_in_composite]
            node_pairs_in_edges_in_composites.extend(node_pairs_in_edges_in_composite)

        # Flatten the list (until now, it's a list of 2-element lists, each pair being the two nodes of an edge)
        return set(itertools.chain.from_iterable(node_pairs_in_edges_in_composite))


def run_main():
    handler = NektarXmlHandler('/home/chris/WorkData/nektar++/actual/bezier/basic_t0.0/tube_bezier_1pt0mesh.xml')
    composite_1 = handler.find_composite_by_id(1)
    print('composite_1:', composite_1)

    nodes_of_first_edge_in_composite_1 = handler.get_nodes_of_edge(composite_1[0])
    print('nodes_of_first_edge_in_composite_1:', nodes_of_first_edge_in_composite_1)

    coordinates_of_first_node_in_edge = handler.get_coordinates_of_node(nodes_of_first_edge_in_composite_1[0])
    print('coordinates_of_first_node_in_edge:', coordinates_of_first_node_in_edge)

    nodes_in_composite = handler.get_nodes_in_composite(1)
    print('nodes in composite:', nodes_in_composite)


if __name__ == '__main__':
    run_main()

