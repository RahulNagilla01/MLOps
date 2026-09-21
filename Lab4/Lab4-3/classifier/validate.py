# T2: Reject unrealistic orders/inputs
def is_valid_input(features):
    # The breast cancer dataset has 30 features
    if len(features) != 30:
        return False

    # Checking specific features for realistic biological boundaries
    # index 0: mean radius, index 1: mean texture, index 3: mean area
    radius_mean = features[0]
    texture_mean = features[1]
    area_mean = features[3]

    if radius_mean <= 0 or radius_mean > 50:
        return False
    if texture_mean <= 0 or texture_mean > 100:
        return False
    if area_mean <= 0 or area_mean > 5000:
        return False

    return True
