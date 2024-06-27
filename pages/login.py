import flet as ft
import json

async def login(page: ft.Page):
    class Auth():
        def __init__(self) -> None:
            self.login = ft.TextField(hint_text='Логин', width=200, color=ft.colors.BLACK,
                hint_style=ft.TextStyle(color=ft.colors.BLACK))
            self.password = ft.TextField(hint_text='Пароль', password=True, 
                can_reveal_password=True, width=200, color=ft.colors.BLACK,
                hint_style=ft.TextStyle(color=ft.colors.BLACK))
            self.type_users = ft.Dropdown(
                width=200,
                label="Уровень доступа",
                options=[
                    ft.dropdown.Option("Мед. работник"),
                    ft.dropdown.Option("Пользователь")],
                color=ft.colors.WHITE,
                label_style=ft.TextStyle(color=ft.colors.BLACK)

            )
            self.register = ft.IconButton(icon=ft.icons.KEY, on_click=lambda x: page.go("/register"))
            self.login_button = ft.ElevatedButton(text='Войти', width=150, on_click=self.login_event)

        async def login_event(self, event):
            if all([self.login.value, self.password.value, self.type_users.value]):
                with open(file='database/users.json', mode='r', encoding='utf-8') as file:
                    data = json.load(file)
                    
                    if data.get(self.login.value):
                        user = data.get(self.login.value)

                        if user['login'] == self.login.value and \
                            user['password'] ==self.password.value and \
                            user['type_user'] == self.type_users.value:

                            if self.type_users.value == 'Мед. работник':
                                page.go('/nurse/home')
                            else:
                                page.go('/user/home')

    auth = Auth()

    login_container = ft.Container(
        content=ft.Column(
            controls=[
                auth.login,
                auth.password,
                auth.type_users,
                ft.Row(controls=[
                    auth.login_button,
                    auth.register
                ], alignment=ft.MainAxisAlignment.CENTER),
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=20,
        ),
        width=400,
        height=300,
        padding=20,
        bgcolor=ft.colors.with_opacity(0.8, ft.colors.GREY),  # Добавляем прозрачность
        border_radius=10,
        shadow=ft.BoxShadow(
            spread_radius=1,
            blur_radius=10,
            color=ft.colors.BLUE_GREY_300,
            offset=ft.Offset(0, 0),
        ), 
        image_src='assets/metap_background/mittap_near.jpg',
        image_fit=ft.ImageFit.COVER,
        image_opacity=0.2
    )

    return ft.Container(
        content=login_container,
        alignment=ft.alignment.center,
    )