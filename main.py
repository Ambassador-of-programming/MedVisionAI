from navigation.route import Router
import flet as ft

async def main(page: ft.Page):
    page.platform = ft.PagePlatform.WINDOWS
    page.theme_mode = ft.ThemeMode.DARK
    page.window.maximized = True
    page.title = 'MedVisionAI'
    page.adaptive = True
    page.padding = 0

    myRouter = Router()
    await myRouter.init(page)

    page.on_route_change = myRouter.route_change
    page.add(
        myRouter.body
    )
    page.go('/login')

if __name__ == "__main__":
    ft.app(target=main, assets_dir='assets')