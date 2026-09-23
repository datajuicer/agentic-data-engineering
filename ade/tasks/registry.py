"""Registry assembled from one module per vertical task."""

from importlib import import_module

from ade.tasks.plugin import TaskPlugin


class TaskRegistry:
    def __init__(self) -> None:
        self._plugins: dict[str, TaskPlugin] = {}
        self._lazy_plugins: dict[str, tuple[str, str]] = {}

    def register(self, plugin: TaskPlugin) -> None:
        if plugin.task_id in self._plugins or plugin.task_id in self._lazy_plugins:
            raise ValueError(f"task already registered: {plugin.task_id}")
        self._plugins[plugin.task_id] = plugin

    def register_lazy(
        self,
        task_id: str,
        module_name: str,
        attribute_name: str = "plugin",
    ) -> None:
        if task_id in self._plugins or task_id in self._lazy_plugins:
            raise ValueError(f"task already registered: {task_id}")
        self._lazy_plugins[task_id] = (module_name, attribute_name)

    def get(self, task_id: str) -> TaskPlugin:
        if task_id not in self._plugins:
            module_name, attribute_name = self._lazy_plugins[task_id]
            plugin = getattr(import_module(module_name), attribute_name)
            if plugin.task_id != task_id:
                raise ValueError(
                    f"task plugin {module_name}.{attribute_name} registered as "
                    f"{plugin.task_id}, expected {task_id}"
                )
            del self._lazy_plugins[task_id]
            self._plugins[task_id] = plugin
        return self._plugins[task_id]

    def task_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._plugins.keys() | self._lazy_plugins.keys()))


def default_task_registry() -> TaskRegistry:
    registry = TaskRegistry()
    for task_id in (
        "evaluation",
        "data_selection",
        "reward_design",
        "curriculum_learning",
    ):
        registry.register_lazy(task_id, f"ade.tasks.{task_id}.plugin")
    return registry
