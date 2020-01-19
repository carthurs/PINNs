import xml.etree.ElementTree as ET

class NektarXmlHandler:
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

    def find_edge_by_id(self, id):
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


def run_main():
    handler = NektarXmlHandler('/home/chris/WorkData/nektar++/actual/bezier/basic_t0.0/tube_bezier_1pt0mesh.xml')
    composite_1 = handler.find_composite_by_id(1)
    print(composite_1)

    nodes_of_first_edge_in_composite_1 = handler.find_edge_by_id(composite_1[0])
    print(nodes_of_first_edge_in_composite_1)

    coordinates_of_first_node_in_edge = handler.get_coordinates_of_node(nodes_of_first_edge_in_composite_1[0])
    print(coordinates_of_first_node_in_edge)




if __name__ == '__main__':
    run_main()

