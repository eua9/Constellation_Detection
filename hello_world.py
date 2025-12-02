#!/usr/bin/env python3
"""
Simple hello world script to verify the environment is set up correctly.
"""

def main():
    print("Hello, World!")
    print("=" * 40)
    print("Constellation Detection System")
    print("Environment Verification")
    print("=" * 40)
    
    # Test imports
    print("\nTesting imports...")
    try:
        import numpy as np
        print("  ✓ numpy")
        
        import scipy  # type: ignore
        print("  ✓ scipy")
        
        import cv2
        print("  ✓ opencv-python")
        
        import skimage  # type: ignore
        print("  ✓ scikit-image")
        
        import matplotlib
        print("  ✓ matplotlib")
        
        print("\nAll dependencies imported successfully!")
        print("\nVersion information:")
        print(f"  NumPy: {np.__version__}")
        print(f"  OpenCV: {cv2.__version__}")
        print(f"  SciKit-Image: {skimage.__version__}")
        
        # Simple numpy test
        arr = np.array([1, 2, 3, 4, 5])
        print(f"\nNumPy test: sum([1,2,3,4,5]) = {arr.sum()}")
        
        print("\n" + "=" * 40)
        print("Environment is ready!")
        print("=" * 40)
        return True
        
    except ImportError as e:
        print(f"\n✗ Import error: {e}")
        print("\nPlease install dependencies with:")
        print("  pip install -r requirements.txt")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

