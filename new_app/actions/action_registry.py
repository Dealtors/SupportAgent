from .change_password import ChangePasswordAction
from .check_ticket_status import CheckTicketStatusAction
from .create_ticket import CreateTicketAction
from .restart_service import RestartServiceAction
from .grant_access import GrantAccessAction

ACTION_REGISTRY = {
    ChangePasswordAction.action_name: ChangePasswordAction(),
    CheckTicketStatusAction.action_name: CheckTicketStatusAction(),
    CreateTicketAction.action_name: CreateTicketAction(),
    RestartServiceAction.action_name: RestartServiceAction(),
    GrantAccessAction.action_name: GrantAccessAction(),
}
