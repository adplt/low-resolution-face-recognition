import argparse
from align_data import PreProcessing

EYES = [i for i in range(36, 48)]


def main(argv):
    align_data = PreProcessing(argv)
    align_data.align_face_youtube_face()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--shapePredictor', required=True, help='Path to facial landmark')
    parser.add_argument('-i', '--image', required=False, help='Path to Input Image')
    args = parser.parse_args()
    main(args)

# Example:
# python ./thesis/model.py --shapePredictor shape_predictor_68_face_landmarks.dat
