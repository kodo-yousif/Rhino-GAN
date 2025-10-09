/* eslint-disable react-refresh/only-export-components */
import { Route, createRoutesFromChildren } from "react-router-dom"

import { lazyPageBuilder } from "./lib/route"

import Error from "@/pages/Error"
import AppLayout from "./containers/AppLayout"

const Home = lazyPageBuilder(() => import("@/pages/Home"))
const VideoPlayer = lazyPageBuilder(() => import("@/pages/video-player"))
const NotFound = lazyPageBuilder(() => import("@/pages/NotFound"))

const routes = createRoutesFromChildren(
  <Route id="root" ErrorBoundary={Error}>
    <Route path="/" Component={AppLayout}>
      <Route index lazy={Home} />
      <Route path="/video" lazy={VideoPlayer} />
    </Route>
    <Route path="*" lazy={NotFound} />
  </Route>
)

export default routes
