import torch
import torch.nn as nn

class ResearchRingSimulator(nn.Module):
    """
    A PyTorch module to simulate a multi-source CT system in a physics laboratory.

    Attributes:
        source_positions (torch.Tensor): The 3D positions of the sources in mm.
        detector_position (torch.Tensor): The position of the detector in mm.
        detector_orientation (torch.Tensor): The orientation of the detector given by three angles in radians.
        object_rotation_angle (float): The rotation angle of the object being scanned.
        swing_gantry_angle (float): The angle of the swinging gantry.
    """

    def __init__(self, source_positions, detector_position, detector_orientation, object_rotation_angle, swing_gantry_angle):
        """
        Initializes the ResearchRingSimulator module.

        Args:
            source_positions (torch.Tensor): A tensor of shape [num_sources, 3] representing the 3D positions of the sources in mm.
            detector_position (torch.Tensor): A tensor of shape [3] representing the position of the detector in mm.
            detector_orientation (torch.Tensor): A tensor of shape [3] representing the orientation of the detector in radians.
            object_rotation_angle (float): The rotation angle of the object in radians.
            swing_gantry_angle (float): The swing angle of the gantry in radians.
        """
        super(ResearchRingSimulator, self).__init__()

        assert source_positions.ndim == 2 and source_positions.shape[1] == 3, "source_positions must be of shape [num_sources, 3]"
        assert detector_position.shape == (3,), "detector_position must be of shape [3]"
        assert detector_orientation.shape == (3,), "detector_orientation must be of shape [3]"

        self.source_positions = source_positions
        self.detector_position = detector_position
        self.detector_orientation = detector_orientation
        self.object_rotation_angle = object_rotation_angle
        self.swing_gantry_angle = swing_gantry_angle


    def _point_forward_projector(self, xyz_object):
        """
        Projects points from object coordinates to the detector plane in a batched fashion.

        Args:
            xyz_object (torch.Tensor): A tensor of shape [n_batch, 3] or [3] representing the (x,y,z) positions of points in the object coordinate system.

        Returns:
            uv_projection (torch.Tensor): A tensor of shape [n_batch, 2] or [2] representing the (u,v) positions of points in the detector coordinate system.
        """
        # Support for both batched and single point inputs
        if xyz_object.dim() == 1:
            xyz_object = xyz_object.unsqueeze(0)  # Add batch dimension if necessary
        assert xyz_object.dim() == 2 and xyz_object.size(1) == 3, "Input position must be of shape [n_batch, 3] or [3]"

        # Define rotation matrices for object and swing gantry frames
        theta = self.object_rotation_angle  # Angle between object and swing gantry frame
        phi = self.swing_gantry_angle  # Angle between lab and swing gantry frame

        roll_angle = self.detector_orientation[0]
        pitch_angle = self.detector_orientation[1]
        yaw_angle = self.detector_orientation[2]

        xyz_sources_lab_frame = self.source_positions

        xyz_detector_swing_frame = self.detector_position
        
        rotation_matrix_object = torch.tensor([
            [torch.cos(theta), -torch.sin(theta), 0],
            [torch.sin(theta), torch.cos(theta), 0],
            [0, 0, 1]
        ]).reshape(1, 3, 3)

       
        rotation_matrix_swing = torch.tensor([
            [torch.cos(phi), -torch.sin(phi), 0],
            [torch.sin(phi), torch.cos(phi), 0],
            [0, 0, 1]
        ]).reshape(1, 3, 3)

        # Apply rotation to compute positions of all sources in object frame
        # Assuming self.source_positions is a tensor of shape [num_sources, 3]
        xyz_sources = torch.bmm(rotation_matrix_object, torch.bmm(rotation_matrix_swing, xyz_sources_lab_frame))

        # for the unit vectors start out with u facing the +x direction and v facing the +z direction
        u_hat = torch.tensor([1, 0, 0])
        v_hat = torch.tensor([0, 0, 1])

        # define three rotation matrices based on the three angles in detector_orientation

        roll_matrix = torch.tensor([
            [1, 0, 0],
            [0, torch.cos(roll_angle), -torch.sin(roll_angle)],
            [0, torch.sin(roll_angle), torch.cos(roll_angle)]
        ])
        

        pitch_matrix = torch.tensor([
            [torch.cos(pitch_angle), 0, torch.sin(pitch_angle)],
            [0, 1, 0],
            [-torch.sin(pitch_angle), 0, torch.cos(pitch_angle)]
        ])

        
        yaw_matrix = torch.tensor([
            [torch.cos(yaw_angle), -torch.sin(yaw_angle), 0],
            [torch.sin(yaw_angle), torch.cos(yaw_angle), 0],
            [0, 0, 1]
        ])

        u_hat = torch.matmul(roll_matrix, u_hat)
        u_hat = torch.matmul(pitch_matrix, u_hat)
        u_hat = torch.matmul(yaw_matrix, u_hat)

        v_hat = torch.matmul(roll_matrix, v_hat)
        v_hat = torch.matmul(pitch_matrix, v_hat)
        v_hat = torch.matmul(yaw_matrix, v_hat)

        # define the detector position in the object frame
        xyz_d = torch.matmul(rotation_matrix_object, xyz_detector_swing_frame)

        # define the projection domain coordinates (u,v) in the object frame, xyz_d is the origin
        u = torch.matmul(rotation_matrix_object, u_hat)
        v = torch.matmul(rotation_matrix_object, v_hat)

        # detector normal vector
        w = torch.cross(u, v)
        
        # Initialize list to hold (u,v) coordinates for each point in batch
        # instead of a list lets initialize a tensor
        uv = torch.zeros((xyz_object.size(0), 2))

        # needed for the intersection point calculation, but only needs to be calculated once
        w_dot_xyz_d_minus_xyz_s = torch.dot(w, xyz_d - xyz_s)

        # Loop over each source position
        for xyz_s in xyz_sources:

            # Loop over each object point
            for i, xyz_o in enumerate(xyz_object):
                
                # the equation of the detector plane is dot(w,r-r_d) = 0, where w is the normal vector to the plane
                # the equation for the line between r_s and r (object) is r = r_s + t(r_o - r_s), where t is a scalar
                # plug in r for the equation of the plane: dot(w, r_s + t(r_o - r_s) - r_d) = 0
                # separate terms depending on t: dot(w, r_s - r_d) + t(dot(w, r_o - r_s)) = 0
                # solve for t: t = -dot(w, r_s - r_d) / dot(w, r_o - r_s)
                # rearrange the negatives: t = dot(w, r_d - r_s) / dot(w, r_o - r_s)
                # solve for t: t = dot(w, r_d - r_s) / dot(w, r_o - r_s)
                # plug t into the equation for the line to get the intersection point: r_i = r_s + t(r_o - r_s) = r_s + (dot(w, r_d - r_s) / dot(w, r_o - r_s)) (r_o - r_s)

                # w_dot_r_d_minus_r_s = torch.dot(w, r_d - r_s) defined outside the loop above
                xyz_o_minus_xyz_s = xyz_o - xyz_s
                r_i = xyz_s + (w_dot_xyz_d_minus_xyz_s / torch.dot(w, xyz_o_minus_xyz_s)) * (xyz_o_minus_xyz_s)

                # Compute the (u,v) coordinates by projecting the intersection point onto the u and v axes
                u_i = torch.dot(r_i - r_d, u)
                v_i = torch.dot(r_i - r_d, v)
                
                # Append the (u,v) coordinates to the list
                uv[i, 0] = u_i
                uv[i, 1] = v_i

        return uv


    def forward(self, x):
        """
        Forward pass of the simulator. This is currently empty and needs to be implemented.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor (to be implemented).
        """
        # TODO: Implement the simulation logic here
        pass














# define a ring of points in an x,y plane 5mm above the origin
n = 100
radius = 50
theta = torch.linspace(0, 2 * torch.pi, n)
x = radius * torch.cos(theta)
y = radius * torch.sin(theta)
z = torch.zeros(n)
r_upper = torch.stack((x, y, z+5), dim=1).transpose(0, 1)
r_lower = torch.stack((x, y, z-5), dim=1).transpose(0, 1)
r = torch.cat((r_upper, r_lower), dim=0)

print('r shape:', r.shape)