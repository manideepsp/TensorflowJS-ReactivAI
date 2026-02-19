declare module '@tensorflow-models/face-landmarks-detection' {
	export type FaceLandmarksDetector = {
		estimateFaces: (
			input: HTMLVideoElement | HTMLImageElement | HTMLCanvasElement,
			config?: unknown
		) => Promise<Array<{
			keypoints: Array<{ x: number; y: number; z?: number; name?: string }>;
			box: {
				xMin: number;
				yMin: number;
				xMax: number;
				yMax: number;
				width: number;
				height: number;
			};
			score?: number;
		}>>;
		dispose: () => void;
	};

	export type MediaPipeFaceMeshTfjsModelConfig = {
		runtime: 'tfjs';
		maxFaces?: number;
		refineLandmarks?: boolean;
	};

	export enum SupportedModels {
		MediaPipeFaceMesh = 'mediapipe_face_mesh',
	}

	export function createDetector(
		model: SupportedModels,
		config: MediaPipeFaceMeshTfjsModelConfig
	): Promise<FaceLandmarksDetector>;
}
