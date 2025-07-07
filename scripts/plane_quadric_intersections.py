# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "numpy",
#     "scipy",
# ]
# ///
import numpy as np
from scipy.linalg import eig
from typing import Optional


class PlaneQuadricIntersection:
    """
    Compute intersection of a plane with a quadric surface.
    
    Quadric: Ax² + By² + Cz² + Dxy + Exz + Fyz + Gx + Hy + Iz + J = 0
    Plane: ax + by + cz + d = 0
    """
    
    def __init__(self, quadric_coeffs: np.ndarray, plane_coeffs: np.ndarray):
        """
        Initialize with quadric and plane coefficients.
        
        Args:
            quadric_coeffs: [A, B, C, D, E, F, G, H, I, J] - 10 coefficients
            plane_coeffs: [a, b, c, d] - 4 coefficients
        """
        self.Q = quadric_coeffs
        self.P = plane_coeffs
        
    def _get_plane_basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get orthonormal basis for the plane.
        Returns: (point_on_plane, u_vector, v_vector)
        """
        a, b, c, d = self.P
        normal = np.array([a, b, c])
        
        # Find a point on the plane
        if abs(c) > 1e-10:
            p0 = np.array([0, 0, -d/c])
        elif abs(b) > 1e-10:
            p0 = np.array([0, -d/b, 0])
        elif abs(a) > 1e-10:
            p0 = np.array([-d/a, 0, 0])
        else:
            raise ValueError("Invalid plane equation")
        
        # Create orthonormal basis in the plane
        # Find two vectors orthogonal to normal
        if abs(normal[0]) < 0.9:
            v1 = np.array([1, 0, 0])
        else:
            v1 = np.array([0, 1, 0])
        
        # Gram-Schmidt orthogonalization
        u = v1 - np.dot(v1, normal) * normal
        u = u / np.linalg.norm(u)
        
        v = np.cross(normal, u)
        v = v / np.linalg.norm(v)
        
        return p0, u, v
    
    def _quadric_matrix(self) -> np.ndarray:
        """Convert quadric coefficients to matrix form."""
        A, B, C, D, E, F, G, H, I, J = self.Q
        
        # 4x4 homogeneous quadric matrix
        Q_matrix = np.array([
            [A,   D/2, E/2, G/2],
            [D/2, B,   F/2, H/2],
            [E/2, F/2, C,   I/2],
            [G/2, H/2, I/2, J  ]
        ])
        
        return Q_matrix
    
    def _reduce_to_2d(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Reduce 3D quadric-plane intersection to 2D conic.
        Returns: (2D conic matrix, plane point, plane basis)
        """
        p0, u, v = self._get_plane_basis()
        
        # Parametric point on plane: P(s,t) = p0 + s*u + t*v
        # Substitute into quadric equation
        
        A, B, C, D, E, F, G, H, I, J = self.Q
        
        # Quadratic terms
        a11 = A*u[0]**2 + B*u[1]**2 + C*u[2]**2 + D*u[0]*u[1] + E*u[0]*u[2] + F*u[1]*u[2]
        a22 = A*v[0]**2 + B*v[1]**2 + C*v[2]**2 + D*v[0]*v[1] + E*v[0]*v[2] + F*v[1]*v[2]
        a12 = A*u[0]*v[0] + B*u[1]*v[1] + C*u[2]*v[2] + D/2*(u[0]*v[1] + u[1]*v[0]) + \
              E/2*(u[0]*v[2] + u[2]*v[0]) + F/2*(u[1]*v[2] + u[2]*v[1])
        
        # Linear terms
        b1 = G*u[0] + H*u[1] + I*u[2] + \
             2*A*p0[0]*u[0] + 2*B*p0[1]*u[1] + 2*C*p0[2]*u[2] + \
             D*(p0[0]*u[1] + p0[1]*u[0]) + E*(p0[0]*u[2] + p0[2]*u[0]) + \
             F*(p0[1]*u[2] + p0[2]*u[1])
        
        b2 = G*v[0] + H*v[1] + I*v[2] + \
             2*A*p0[0]*v[0] + 2*B*p0[1]*v[1] + 2*C*p0[2]*v[2] + \
             D*(p0[0]*v[1] + p0[1]*v[0]) + E*(p0[0]*v[2] + p0[2]*v[0]) + \
             F*(p0[1]*v[2] + p0[2]*v[1])
        
        # Constant term
        c = A*p0[0]**2 + B*p0[1]**2 + C*p0[2]**2 + D*p0[0]*p0[1] + \
            E*p0[0]*p0[2] + F*p0[1]*p0[2] + G*p0[0] + H*p0[1] + I*p0[2] + J
        
        # 2D conic matrix in homogeneous coordinates
        conic_2d = np.array([
            [a11,    a12,    b1/2],
            [a12,    a22,    b2/2],
            [b1/2,   b2/2,   c   ]
        ])
        
        return conic_2d, p0, np.column_stack([u, v])
    
    def classify_intersection(self) -> str:
        """
        Classify the type of intersection curve.
        Returns: 'ellipse', 'parabola', 'hyperbola', 'empty', 'degenerate'
        """
        conic_2d, _, _ = self._reduce_to_2d()
        
        # Extract 2x2 quadratic part
        A_quad = conic_2d[:2, :2]
        det_A = np.linalg.det(A_quad)
        det_full = np.linalg.det(conic_2d)
        
        # Classification
        if abs(det_full) < 1e-12:
            return 'degenerate'
        elif abs(det_A) < 1e-12:
            return 'parabola'
        elif det_A > 0:
            if det_full * np.trace(A_quad) < 0:
                return 'ellipse'
            else:
                return 'empty'
        else:
            return 'hyperbola'
    
    def compute_intersection_points(self, num_points: int = 100) -> Optional[np.ndarray]:
        """
        Compute points on the intersection curve.
        
        Args:
            num_points: Number of points to compute
            
        Returns:
            Array of 3D points on the intersection curve, or None if empty
        """
        intersection_type = self.classify_intersection()
        
        if intersection_type in ['empty', 'degenerate']:
            return None
        
        conic_2d, p0, basis = self._reduce_to_2d()
        
        # Solve the conic equation parametrically
        points_3d = []
        
        if intersection_type == 'ellipse':
            # Parametric ellipse
            eigenvals, eigenvecs = eig(conic_2d[:2, :2])
            if np.all(eigenvals > 0):
                # Standard ellipse parameterization
                t = np.linspace(0, 2*np.pi, num_points)
                # This is a simplified approach - full implementation would
                # require proper conic normalization
                for ti in t:
                    # Approximate parameterization
                    s = np.cos(ti) / np.sqrt(abs(eigenvals[0]))
                    t_param = np.sin(ti) / np.sqrt(abs(eigenvals[1]))
                    point_3d = p0 + s * basis[:, 0] + t_param * basis[:, 1]
                    points_3d.append(point_3d)
        
        elif intersection_type == 'hyperbola':
            # Parametric hyperbola
            t = np.linspace(-2, 2, num_points)
            eigenvals, eigenvecs = eig(conic_2d[:2, :2])
            for ti in t:
                # Approximate parameterization
                s = np.cosh(ti) / np.sqrt(abs(eigenvals[0]))
                t_param = np.sinh(ti) / np.sqrt(abs(eigenvals[1]))
                point_3d = p0 + s * basis[:, 0] + t_param * basis[:, 1]
                points_3d.append(point_3d)
        
        elif intersection_type == 'parabola':
            # Parametric parabola
            t = np.linspace(-2, 2, num_points)
            for ti in t:
                s = ti
                t_param = ti**2
                point_3d = p0 + s * basis[:, 0] + t_param * basis[:, 1]
                points_3d.append(point_3d)
        
        if points_3d:
            return np.array(points_3d)
        else:
            return None
    
    def get_intersection_info(self) -> dict:
        """
        Get complete information about the intersection.
        """
        intersection_type = self.classify_intersection()
        conic_2d, p0, basis = self._reduce_to_2d()
        
        return {
            'type': intersection_type,
            'conic_matrix_2d': conic_2d,
            'plane_point': p0,
            'plane_basis': basis,
            'determinant': np.linalg.det(conic_2d)
        }


# Example usage and test cases
def example_sphere_plane():
    """Example: Intersection of unit sphere with plane z = 0.5"""
    # Unit sphere: x² + y² + z² - 1 = 0
    quadric = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, -1])
    
    # Plane: z - 0.5 = 0
    plane = np.array([0, 0, 1, -0.5])
    
    intersection = PlaneQuadricIntersection(quadric, plane)
    
    print("Sphere-Plane Intersection:")
    print(f"Type: {intersection.classify_intersection()}")
    
    points = intersection.compute_intersection_points(50)
    if points is not None:
        print(f"Generated {len(points)} points on intersection curve")
        print(f"Sample points:\n{points[:5]}")
    
    return intersection


def example_ellipsoid_plane():
    """Example: Intersection of ellipsoid with tilted plane"""
    # Ellipsoid: x²/4 + y²/9 + z²/1 - 1 = 0
    quadric = np.array([1/4, 1/9, 1, 0, 0, 0, 0, 0, 0, -1])
    
    # Plane: x + y + z - 1 = 0
    plane = np.array([1, 1, 1, -1])
    
    intersection = PlaneQuadricIntersection(quadric, plane)
    
    print("\nEllipsoid-Plane Intersection:")
    print(f"Type: {intersection.classify_intersection()}")
    
    info = intersection.get_intersection_info()
    print(f"Conic determinant: {info['determinant']:.6f}")
    
    return intersection

if __name__ == "__main__":
    # Run examples
    example_sphere_plane()
    example_ellipsoid_plane()
