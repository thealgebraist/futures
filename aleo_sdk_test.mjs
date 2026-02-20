import { Account } from '@aleohq/sdk';

try {
    const account = new Account();
    console.log("--- Aleo SDK Programmatic Check ---");
    // Standard Aleo SDK objects usually have a to_string() method
    console.log("New Address:", account.address().to_string());
    console.log("View Key:", account.viewKey().to_string());
    console.log("Private Key:", account.privateKey().to_string());
    console.log("Status: SDK initialized and functional.");
} catch (error) {
    console.error("SDK Error:", error);
}
