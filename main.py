from utils import (read_video,
                   save_video,
                   measure_distance,
                   convert_pixel_distance_to_meters)

from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt
import cv2


def main():
    # Reading video
    input_video_path = "input_videos/input_video.mp4"
    video_frames = read_video(input_video_path)

    # Detecting Players and ball
    player_tracker = PlayerTracker(model_path="yolov8x")
    ball_tracker = BallTracker(model_path="models/yolo5_last.pt")
    player_detections = player_tracker.detect_frames(
        video_frames, read_from_stub=True, stub_path="./tracker_stubs/player_detections.pkl")

    ball_detections = ball_tracker.detect_frames(
        video_frames, read_from_stub=True, stub_path="./tracker_stubs/ball_detections.pkl")

    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    # Court Line Detection
    court_model_path = "models/keypoints_model.pth"
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])

    # Choose players
    player_detections = player_tracker.choose_and_filter_players(
        court_keypoints, player_detections)

    # Init Mini Court
    mini_court = MiniCourt(video_frames[0])

    # Detect ball shots
    ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_detections)
    # print(ball_shot_frames)

    # Convert positions to mini court positions
    player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(
        player_detections, ball_detections, court_keypoints)

    # Shot speed and player speed
    for ball_shot_ind in range(len(ball_shot_frames)-1):
        start_frame = ball_shot_frames[ball_shot_ind]
        end_frame = ball_shot_frames[ball_shot_ind+1]
        # 24 is the frame rate of the video
        ball_shot_time_in_seconds = (end_frame-start_frame)/24

        # get distance covered by the ball
        distance_covered_by_ball_pixels = measure_distance(
            ball_mini_court_detections[start_frame][1], ball_mini_court_detections[end_frame][1])
        distance_covered_by_ball_meters = mini_court.convert_pixel_distance_to_meters(
            distance_covered_by_ball_pixels)

        # speed of the ball shot in km/h
        speed_of_ball_shot = distance_covered_by_ball_meters/ball_shot_time_in_seconds * 3.6

        # player who shot the ball
        player_positions = player_mini_court_detections[start_frame]
        player_shot_ball = min(player_positions.keys(), key=lambda player_id: measure_distance(player_positions[player_id],
                                                                                               ball_mini_court_detections[start_frame][1]))

        # opponent player speed

    # Draw Output

    # Draw player bounding boxes
    output_video_frames = player_tracker.draw_bboxes(
        video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(
        output_video_frames, ball_detections)

    # Draw court keypoints
    output_video_frames = court_line_detector.draw_keypoints_on_video(
        output_video_frames, court_keypoints)

    # Draw Mini Court
    output_video_frames = mini_court.draw_mini_court(output_video_frames)
    output_video_frames = mini_court.draw_points_on_mini_court(
        output_video_frames, player_mini_court_detections)
    output_video_frames = mini_court.draw_points_on_mini_court(
        output_video_frames, ball_mini_court_detections, color=(0, 255, 255))

    # Draw frame number on top left corner
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame {i}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    save_video(output_video_frames, "output_videos/output_video.avi")


if __name__ == "__main__":
    main()
