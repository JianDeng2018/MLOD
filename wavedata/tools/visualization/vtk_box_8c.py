import vtk
import numpy as np

from wavedata.tools.visualization.vtk_text_labels import VtkTextLabels


class VtkBox8c:
    """
    VtkBox8c (i.e. box 8 corners format) to display corners as a box3D
    """

    def __init__(self):

        # VTK Data
        self.vtk_points = vtk.vtkPoints()
        self.vtk_poly_data = vtk.vtkPolyData()
        self.vtk_text_labels = VtkTextLabels()

        # Data Mapper
        self.vtk_data_mapper = vtk.vtkPolyDataMapper()

        self.vtk_actor = vtk.vtkActor()
        self.vtk_actor.GetProperty().SetRepresentationToWireframe()

    def set_objects(self, corners, colour=None):

        self.create_point_corners(corners)
        lines = self.create_lines()
        self.vtk_poly_data.SetPoints(self.vtk_points)
        self.vtk_poly_data.SetLines(lines)

        # Setup the colours array
        colours = vtk.vtkUnsignedCharArray()
        colours.SetNumberOfComponents(3)
        colours.SetName("Colours")
        if colour is None:
            colour = [0, 255, 0]
        # Add the colours we created to the colours array
        for i in range(12):
            colours.InsertNextTypedTuple(colour)

        self.vtk_poly_data.GetCellData().SetScalars(colours)
        self.vtk_data_mapper.SetInputData(self.vtk_poly_data)
        self.vtk_actor.SetMapper(self.vtk_data_mapper)

    def create_point_corners(self, corners):

        # Draw each line of the cube
        p1 = corners[:, 0]
        p2 = corners[:, 1]
        p3 = corners[:, 2]
        p4 = corners[:, 3]

        p5 = corners[:, 4]
        p6 = corners[:, 5]
        p7 = corners[:, 6]
        p8 = corners[:, 7]

        self.vtk_points.InsertNextPoint(p1)
        self.vtk_points.InsertNextPoint(p2)
        self.vtk_points.InsertNextPoint(p3)
        self.vtk_points.InsertNextPoint(p4)
        self.vtk_points.InsertNextPoint(p5)
        self.vtk_points.InsertNextPoint(p6)
        self.vtk_points.InsertNextPoint(p7)
        self.vtk_points.InsertNextPoint(p8)

        p_names = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8']

        corners = np.swapaxes(corners, 1, 0)
        self.vtk_text_labels.set_text_labels(corners, p_names)

    def create_lines(self):

        ############################
        # Create the box bottom face
        ############################

        line1 = vtk.vtkLine()
        line1.GetPointIds().SetId(0, 0)
        line1.GetPointIds().SetId(1, 1)

        line2 = vtk.vtkLine()
        line2.GetPointIds().SetId(0, 1)
        line2.GetPointIds().SetId(1, 2)

        line3 = vtk.vtkLine()
        line3.GetPointIds().SetId(0, 2)
        line3.GetPointIds().SetId(1, 3)

        line4 = vtk.vtkLine()
        line4.GetPointIds().SetId(0, 0)
        line4.GetPointIds().SetId(1, 3)

        ############################
        # Create the box top face
        ############################
        line5 = vtk.vtkLine()
        line5.GetPointIds().SetId(0, 4)
        line5.GetPointIds().SetId(1, 5)

        line6 = vtk.vtkLine()
        line6.GetPointIds().SetId(0, 5)
        line6.GetPointIds().SetId(1, 6)

        line7 = vtk.vtkLine()
        line7.GetPointIds().SetId(0, 6)
        line7.GetPointIds().SetId(1, 7)

        line8 = vtk.vtkLine()
        line8.GetPointIds().SetId(0, 7)
        line8.GetPointIds().SetId(1, 4)

        ############################
        # Create the box edges
        ############################
        line9 = vtk.vtkLine()
        line9.GetPointIds().SetId(0, 0)
        line9.GetPointIds().SetId(1, 4)

        line10 = vtk.vtkLine()
        line10.GetPointIds().SetId(0, 1)
        line10.GetPointIds().SetId(1, 5)

        line11 = vtk.vtkLine()
        line11.GetPointIds().SetId(0, 2)
        line11.GetPointIds().SetId(1, 6)

        line12 = vtk.vtkLine()
        line12.GetPointIds().SetId(0, 3)
        line12.GetPointIds().SetId(1, 7)

        # Create a cell array to store the lines in and
        # add the lines to it
        lines = vtk.vtkCellArray()
        lines.InsertNextCell(line1)
        lines.InsertNextCell(line2)
        lines.InsertNextCell(line3)
        lines.InsertNextCell(line4)
        lines.InsertNextCell(line5)
        lines.InsertNextCell(line6)
        lines.InsertNextCell(line7)
        lines.InsertNextCell(line8)
        lines.InsertNextCell(line9)
        lines.InsertNextCell(line10)
        lines.InsertNextCell(line11)
        lines.InsertNextCell(line12)

        return lines
