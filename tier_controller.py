# ============================================================
# ADAPTIVE VIDEO TIER CONTROLLER
# ============================================================


def get_tier(activity):

    if activity == "IDLE":

        return {
            "tier": 0,
            "width": 320,
            "height": 240,
            "fps": 5,
            "jpeg_quality": 40
        }


    elif activity == "MOTION":

        return {
            "tier": 1,
            "width": 480,
            "height": 360,
            "fps": 8,
            "jpeg_quality": 50
        }


    elif activity == "PERSON":

        return {
            "tier": 2,
            "width": 640,
            "height": 480,
            "fps": 10,
            "jpeg_quality": 60
        }


    elif activity == "LOITERING":

        return {
            "tier": 3,
            "width": 640,
            "height": 480,
            "fps": 15,
            "jpeg_quality": 70
        }


    else:

        return {
            "tier": 0,
            "width": 320,
            "height": 240,
            "fps": 5,
            "jpeg_quality": 40
        }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    activities = [
        "IDLE",
        "MOTION",
        "PERSON",
        "LOITERING"
    ]


    for activity in activities:

        settings = get_tier(activity)

        print()
        print("Activity:", activity)
        print("Tier:", settings["tier"])
        print(
            "Resolution:",
            f'{settings["width"]}x{settings["height"]}'
        )
        print("FPS:", settings["fps"])
        print(
            "JPEG Quality:",
            settings["jpeg_quality"]
        )