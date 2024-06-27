import flet as ft
import json

async def register(page: ft.Page):
    class Auth():
        def __init__(self) -> None:
            self.login = ft.TextField(hint_text='Логин', width=200, color=ft.colors.BLACK,
                hint_style=ft.TextStyle(color=ft.colors.BLACK))
            
            self.password = ft.TextField(hint_text='Пароль', password=True, 
                width=200, color=ft.colors.BLACK,
                hint_style=ft.TextStyle(color=ft.colors.BLACK))
            
            self.fio = ft.TextField(hint_text='ФИО', width=200, color=ft.colors.BLACK,
                hint_style=ft.TextStyle(color=ft.colors.BLACK))
            
            self.date_of_birth = ft.TextField(hint_text='Дата рождения', password=True, 
                width=200, color=ft.colors.BLACK,
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
            self.login_home = ft.IconButton(icon=ft.icons.HOME, on_click=lambda x: page.go("/login"))
            self.login_button = ft.ElevatedButton(text='Зарегаться', width=150, on_click=self.register_event)

        async def register_event(self, event):
            if all([self.login.value, self.password.value, self.fio.value, self.date_of_birth.value,
                self.type_users.value]):
                new_user = {
                    "login": self.login.value.strip(),
                    "password": self.password.value.strip(),
                    "fio": self.fio.value.strip(),
                    "date_of_birth": self.date_of_birth.value.strip(),
                    "type_user": self.type_users.value.strip()
                }
                await self.save_to_json(new_user)

        async def save_to_json(self, new_user):
            file_path = 'database/users.json'
            try:
                # Чтение существующих данных
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    if content:
                        data = json.loads(content)
                    else:
                        data = {}
            except FileNotFoundError:
                # Если файл не существует, создаем новый словарь
                data = {}
            except json.JSONDecodeError:
                # Если файл содержит некорректный JSON
                print(f"Ошибка при чтении файла {file_path}. Создание нового словаря.")
                data = {}

            # Добавление нового пользователя
            # Используем логин как ключ
            data[new_user['login']] = new_user

            # Запись обновленных данных обратно в файл
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(data, file, ensure_ascii=False, indent=4)

    auth = Auth()

    login_container = ft.Container(
        content=ft.Column(
            controls=[
                auth.login,
                auth.password,
                auth.fio,
                auth.date_of_birth,
                auth.type_users,
                ft.Row(controls=[
                    auth.login_button,
                    auth.login_home,
                ], alignment=ft.MainAxisAlignment.CENTER),
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=20,
        ),
        width=400,
        height=450,
        padding=20,
        bgcolor=ft.colors.with_opacity(0.8, ft.colors.GREY),  # Добавляем прозрачность
        border_radius=10,
        shadow=ft.BoxShadow(
            spread_radius=1,
            blur_radius=10,
            color=ft.colors.BLUE_GREY_300,
            offset=ft.Offset(0, 0),
        ), 
        image_src='assets/metap_background/metap_away.jpg',
        image_fit=ft.ImageFit.COVER,
        image_opacity=0.2
    )

    return ft.Container(
        content=login_container,
        alignment=ft.alignment.center,
        # expand=True,
        # bgcolor=ft.colors.BLUE_GREY_100,  # Цвет фона всей страницы
    )