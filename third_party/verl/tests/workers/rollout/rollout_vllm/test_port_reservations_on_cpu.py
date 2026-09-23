import asyncio
from unittest.mock import Mock

from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer


class PortReservationsReleased(Exception):
    pass


def test_release_port_reservations_closes_each_socket_once():
    server = object.__new__(vLLMHttpServer)
    reserved_sockets = [Mock(), Mock(), Mock()]
    server._master_sock, server._dp_rpc_sock, server._dp_master_sock = reserved_sockets

    server._release_port_reservations()
    server._release_port_reservations()

    for reserved_socket in reserved_sockets:
        reserved_socket.close.assert_called_once_with()
    assert server._master_sock is None
    assert server._dp_rpc_sock is None
    assert server._dp_master_sock is None


def test_release_port_reservations_handles_non_master_server():
    server = object.__new__(vLLMHttpServer)

    server._release_port_reservations()

    assert server._master_sock is None
    assert server._dp_rpc_sock is None
    assert server._dp_master_sock is None


def test_run_server_releases_ports_before_engine_initialization():
    server = object.__new__(vLLMHttpServer)
    server._release_port_reservations = Mock(side_effect=PortReservationsReleased)

    try:
        asyncio.run(server.run_server(object()))
    except PortReservationsReleased:
        pass
    else:
        raise AssertionError("run_server did not release reserved ports before engine initialization")

    server._release_port_reservations.assert_called_once_with()
