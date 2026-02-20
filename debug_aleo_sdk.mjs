import { Account } from '@aleohq/sdk';

const account = new Account();
console.log("Available Account methods:", Object.getOwnPropertyNames(Object.getPrototypeOf(account)));
console.log("Account object keys:", Object.keys(account));
