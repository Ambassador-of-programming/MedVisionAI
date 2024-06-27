import flet as ft

async def home_nurse(page: ft.Page):
    class Auth():
        def __init__(self) -> None:
            self.download_file = ft.ElevatedButton(
                    "Pick files",
                    icon=ft.icons.UPLOAD_FILE,
                    # on_click=lambda _: pick_files_dialog.pick_files(
                    #     allow_multiple=True
                    # ),
                    width=150,
                    bgcolor=ft.colors.BLUE
                )
            
            self.result_text = ft.Text(value='Результат:', color='black', width=100)
            self.result_icon_none = ft.Icon(name=ft.icons.QUESTION_MARK_SHARP, color=ft.colors.BLACK)
            self.result_icon_true = ft.Icon(name=ft.icons.GPP_GOOD_SHARP, color='green')
            self.result_icon_bad = ft.Icon(name=ft.icons.GPP_BAD_SHARP, color='red')
            self.button_check = ft.ElevatedButton(text='Отправить на анализ', on_click=None,
                width=200,
            )
            self.home = ft.IconButton(icon=ft.icons.KEY_OFF, on_click=lambda x: page.go('/login'), icon_color=ft.colors.RED_500)
            self.exit = ft.IconButton(icon=ft.icons.EXIT_TO_APP, on_click=lambda x: page.window.close(), icon_color=ft.colors.RED)


    auth = Auth()

    login_container = ft.Container(
        content=ft.Column(
            controls=[
                auth.download_file,
                ft.Row(controls=[
                    auth.result_text,
                    auth.result_icon_none
                ], alignment=ft.MainAxisAlignment.CENTER),
                auth.button_check,

                ft.Row(controls=[
                    auth.home,
                    auth.exit
                ], alignment=ft.MainAxisAlignment.END),
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=20,
        ),
        width=400,
        height=250,
        padding=20,
        bgcolor=ft.colors.with_opacity(0.8, ft.colors.GREY),  # Добавляем прозрачность
        border_radius=10,
        shadow=ft.BoxShadow(
            spread_radius=1,
            blur_radius=10,
            color=ft.colors.BLUE_GREY_300,
            offset=ft.Offset(0, 0),
        ), 
        image_src='assets/metap_background/imgonline-com-ua-GifAnimationTwoPic-v7Y3POBFBxVu.gif',
        image_fit=ft.ImageFit.COVER,
        image_opacity=0.2
    )

    return ft.Container(
        content=login_container,
        alignment=ft.alignment.center,
    )