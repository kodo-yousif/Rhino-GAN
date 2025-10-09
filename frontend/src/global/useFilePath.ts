import { create } from "zustand"
import { persist } from "zustand/middleware"

type State = {
  filePath: string
  setFilePath: (path: string) => void
}

export const useFilePath = create<State>()(
  persist(
    (set) => ({
      filePath: "C:\\projects\\Research\\nose\\images",
      setFilePath: (path: string) => set({ filePath: path }),
    }),
    {
      name: "FilePath",
    }
  )
)
