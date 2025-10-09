import { create } from "zustand"
import { persist } from "zustand/middleware"

export const IMAGE_ROWS = 512
export const IMAGE_COLS = 512

export type Landmark = [number, number, number]

export type RawNode = {
  key: string
  title: string
  fullPath: string
  children?: RawNode[]
  type: "video" | "image" | "folder"
}

export type SegmentationData = {
  segmentedImage: number[][]
  segmentedImageRef: number[][]
  COLOR_MAP: [number, number, number][]
  regions: {
    nose: number
    skin: number
  }
} | null

interface IPaintArray {
  row: number
  draw: boolean
  skin?: number
  column: number
  newRegion: number
  paintSize: number
  image: number[][]
  refImage?: number[][]
}
function paintArray({
  row,
  draw,
  skin,
  column,
  image,
  refImage,
  paintSize,
  newRegion,
}: IPaintArray) {
  const rows = image.length
  const cols = image[0].length
  const r = Math.max(0, Math.floor(paintSize)) // radius in cells
  if (r === 0) {
    image[row][column] = newRegion
    return
  }

  const rSquared = r * r

  for (let dy = -r; dy <= r; dy++) {
    const y = row + dy
    if (y < 0 || y >= rows) continue

    for (let dx = -r; dx <= r; dx++) {
      const x = column + dx
      if (x < 0 || x >= cols) continue

      if (dx * dx + dy * dy <= rSquared) {
        if (draw) image[y][x] = newRegion
        else if (refImage) {
          if (refImage[y][x] === newRegion && skin) image[y][x] = skin
          else image[y][x] = refImage[y][x]
        }
      }
    }
  }
}

export type ImageNodesState = {
  noseStyle: string
  showNodes: boolean[]
  showSegmentation: boolean[]
  showNodeLandmarks: boolean[]
  nodes: (RawNode | undefined)[]
  segmentationDatas: SegmentationData[]
  nodeLandmarksInfo: { referenceLandmarks: Landmark[]; landmarks: Landmark[] }[]
  modules: {
    noseStyle: boolean
    landmarks: boolean
    segmentation: boolean
    doStyle: boolean
  }
  drawSegmentationByIndex: (
    idx: number,
    row: number,
    column: number,
    newRegion: number,
    paintSize: number
  ) => void
  undrawSegmentationByIndex: (
    idx: number,
    row: number,
    column: number,
    paintSize: number
  ) => void
  setNoseStyle: (newNoseStyle: string) => void
  toggleModuleByName: (name: keyof ImageNodesState["modules"]) => void
  toggleLandmarkByIndex: (idx: number) => void
  toggleSegmentationByIndex: (idx: number) => void
  setShowNodeByIndex: (idx: number, boolean: boolean) => void
  setNodeByIndex: (idx: number, newNode: RawNode | undefined) => void
  setSegmentationDataByIndex: (
    idx: number,
    newSegmentationData: SegmentationData
  ) => void
  setNodeLandmarksInfoByIndex: (
    idx: number,
    name: keyof ImageNodesState["nodeLandmarksInfo"][0],
    landmarks: Landmark[]
  ) => void
}

export const useImageNodes = create<ImageNodesState>()(
  persist(
    (set, get) => ({
      noseStyle: "self",
      setNoseStyle: (newNoseStyle) => set({ noseStyle: newNoseStyle }),
      drawSegmentationByIndex: (idx, row, column, newRegion, paintSize) => {
        const currentSegmentationData = get().segmentationDatas[idx]
        if (!currentSegmentationData) return

        const segmentedImage: number[][] = JSON.parse(
          JSON.stringify(currentSegmentationData.segmentedImage)
        )

        paintArray({
          row,
          column,
          newRegion,
          paintSize,
          draw: true,
          image: segmentedImage,
        })

        set({
          // @ts-ignore
          segmentationDatas: get().segmentationDatas.map(
            (data: SegmentationData, i) =>
              i === idx ? { ...data, segmentedImage } : data
          ),
        })
      },
      undrawSegmentationByIndex: (idx, row, column, paintSize) => {
        const currentSegmentationData = get().segmentationDatas[idx]
        if (!currentSegmentationData) return

        const segmentedImage: number[][] = JSON.parse(
          JSON.stringify(currentSegmentationData.segmentedImage)
        )

        paintArray({
          row,
          column,
          paintSize,
          draw: false,
          image: segmentedImage,
          skin: currentSegmentationData.regions.skin,
          newRegion: currentSegmentationData.regions.nose,
          refImage: currentSegmentationData.segmentedImageRef,
        })

        set({
          // @ts-ignore
          segmentationDatas: get().segmentationDatas.map(
            (data: SegmentationData, i) =>
              i === idx ? { ...data, segmentedImage } : data
          ),
        })
      },
      modules: {
        doStyle: true,
        noseStyle: true,
        landmarks: true,
        segmentation: true,
      },
      segmentationDatas: [null, null],
      nodeLandmarksInfo: [
        { referenceLandmarks: [], landmarks: [] },
        { referenceLandmarks: [], landmarks: [] },
      ],
      nodes: [undefined, undefined],
      showNodes: [true, false],
      showNodeLandmarks: [true, true],
      showSegmentation: [true, false],
      toggleModuleByName: (name) =>
        set({
          modules: { ...get().modules, [name]: !get().modules[name] },
        }),
      setShowNodeByIndex: (nodeIdx, showNewNode) =>
        set({
          showNodes: get().showNodes.map((showCurrentNode, currentIdx) =>
            nodeIdx === currentIdx ? showNewNode : showCurrentNode
          ),
        }),
      setSegmentationDataByIndex: (idx, newSegmentationData) =>
        set({
          segmentationDatas: get().segmentationDatas.map(
            (currentSegmentationData, currentIdx) =>
              idx === currentIdx ? newSegmentationData : currentSegmentationData
          ),
        }),
      setNodeLandmarksInfoByIndex: (idx, name, newLandmarks) =>
        set({
          nodeLandmarksInfo: get().nodeLandmarksInfo.map(
            (currentLandmarks, currentIdx) =>
              idx === currentIdx
                ? { ...currentLandmarks, [name]: newLandmarks }
                : currentLandmarks
          ),
        }),
      setNodeByIndex: (nodeIdx, newNode) => {
        const payload: Partial<ImageNodesState> = {}

        payload.segmentationDatas = [null, null]
        payload.nodeLandmarksInfo = get().nodeLandmarksInfo.map(
          (currentLandmarks, currentIdx) =>
            nodeIdx === currentIdx
              ? { referenceLandmarks: [], landmarks: [] }
              : currentLandmarks
        )

        payload.nodes = get().nodes.map((currentNode, currentIdx) =>
          nodeIdx === currentIdx ? newNode : currentNode
        )

        payload.noseStyle = "self"

        set({
          ...payload,
        })
      },
      toggleLandmarkByIndex: (imageIdx: number) =>
        set({
          showNodeLandmarks: get().showNodeLandmarks.map(
            (showState, stateIdx) =>
              stateIdx === imageIdx ? !showState : showState
          ),
        }),
      toggleSegmentationByIndex: (imageIdx: number) =>
        set({
          showSegmentation: get().showSegmentation.map((showState, stateIdx) =>
            stateIdx === imageIdx ? !showState : showState
          ),
        }),
    }),
    {
      name: "ImageNodeStates",
    }
  )
)
