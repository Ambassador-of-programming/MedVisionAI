from pages.login import login
from pages.register import register
from pages.type_user.medic.nurse_home import home_nurse
from pages.type_user.user.user_home import home_user
import flet as ft
import random

class Router:
    async def init(self, page: ft.Page):
        self.routes = {
            '/login': await login(page),
            '/register': await register(page),
            '/nurse/home': await home_nurse(page),
            '/user/home': await home_user(page),
        }
        
        self.body = ft.Container(
            expand=True,
            image_src=await self.random_background(),
            image_fit=ft.ImageFit.COVER,
        )
    
    async def random_background(self):
        choice = random.choice([
            'assets/neurons_background/1.gif', 'assets/neurons_background/2.gif',
            'assets/neurons_background/3.gif', 'assets/neurons_background/4.gif', 
            'assets/neurons_background/5.gif','assets/neurons_background/6.gif', 
            'assets/neurons_background/7.gif',])
        return choice

    async def route_change(self, route):
        self.body.content = self.routes[route.route]
        self.body.update()