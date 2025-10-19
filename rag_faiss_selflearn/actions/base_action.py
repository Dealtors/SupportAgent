class BaseAction:
    """
    Базовый класс для всех действий. 
    Каждое действие должно реализовать метод execute(data: dict).
    """
    action_name = "base"

    def execute(self, data: dict) -> dict:
        raise NotImplementedError("Метод execute() должен быть реализован в дочернем классе")
