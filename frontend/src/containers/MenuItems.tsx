import type { MenuProps } from "antd"
import { HomeFilled } from "@ant-design/icons"

type MenuItem = Required<MenuProps>["items"][number]

function getItem(
  label: React.ReactNode,
  key: React.Key,
  icon?: React.ReactNode,
  children?: MenuItem[]
): MenuItem {
  return {
    key,
    icon,
    label,
    children,
    i18n: label,
  } as MenuItem
}

const menuItems: MenuItem[] = [getItem("home", "/", <HomeFilled />)]

export default menuItems
