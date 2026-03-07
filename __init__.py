from .run import MatAnyoneVideo, MatAnyone2Video, SolidColorBatched

NODE_CLASS_MAPPINGS = {
    "MatAnyone": MatAnyoneVideo,
    "MatAnyone2": MatAnyone2Video,
    "SolidColorBatched": SolidColorBatched,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MatAnyone": "MatAnyone",
    "MatAnyone2": "MatAnyone2",
    "SolidColorBatched": "Solid Color Batched",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
