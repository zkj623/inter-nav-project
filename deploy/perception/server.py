"""Run the release GroundingDINO, MobileSAM and Qwen-VL service interfaces."""


def main():
    from grutopia.demo.serve_semantic_perception import main as serve
    serve()
