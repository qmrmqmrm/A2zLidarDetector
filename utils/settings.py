import os
import sys

# add current path
sub_package_path = os.path.dirname(os.path.abspath(__file__))
if sub_package_path not in sys.path:
    sys.path.append(sub_package_path)

# add current path
package_path = os.path.dirname(sub_package_path)
if package_path not in sys.path:
    sys.path.append(package_path)
# print("syspath", sys.path)
